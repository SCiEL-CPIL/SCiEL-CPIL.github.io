"""
Strategy: Smooth Hyperbolic Tangent Yield Function.
The current program uses localized Gaussian RBF bumps. While flexible, RBFs
can struggle with extrapolation or create "bumpy" derivatives if not carefully
tuned, and they don't inherently respect the global structure of a stress-strain
curve (elastic -> yield -> hardening).

This new strategy uses a single, smooth, global functional form based on the
hyperbolic tangent (tanh). This function naturally models a transition from
one linear regime to another:
- For small strain, tanh(x) ≈ x, giving the elastic response.
- For large strain, tanh(x) ≈ 1, giving the plastic offset/hardening.

The model is:
sigma = (E0 + E1*t) * s + (S0 + S1*t) * tanh( (s - s_y) / w )

Where:
- (E0 + E1*t) * s is the linear elastic term.
- (S0 + S1*t) is the amplitude of the yield/hardening offset.
- tanh( (s - s_y) / w ) is the smooth transition function.
  - s_y is the yield strain (center of the transition).
  - w is the width of the transition (inverse of the sharpness).

This uses 5 parameters for the core physics. We have 10 available.
To improve fit quality, we can add:
1. A quadratic elastic term (to account for slight non-linearity in elasticity or measurement noise).
2. A quadratic plastic hardening term (to account for non-linear hardening after yield).
3. A constant offset (to account for zero-point errors).
4. A temperature-dependent shift in yield strain.

Let's define a richer form with 10 parameters:
p0: Elastic Modulus E0
p1: Temp coeff for E (E1)
p2: Yield Strain center (s_y)
p3: Temp coeff for s_y (s_y1)
p4: Transition Width (w)
p5: Yield Amplitude S0
p6: Temp coeff for S0 (S1)
p7: Quadratic Elastic Coeff (E2)
p8: Quadratic Hardening Coeff (S2)
p9: Constant Offset (C)

sigma = (p0 + p1*t) * s + p7 * s^2 
        + (p5 + p6*t) * tanh( (s - (p2 + p3*t)) / w )
        + p8 * s * tanh( (s - (p2 + p3*t)) / w )  # Optional: Hardening slope change?
        + p9

Actually, a simpler and more robust extension is to allow the plastic region to have its own slope.
Standard bilinear model:
Elastic: E * s
Plastic: E * s + S * tanh(...) is not quite right for slope.
Better:
sigma = E * s + S * tanh( (s - sy) / w )
This has slope E in elastic, and slope E + S/w at yield, then E again? No.
Derivative of tanh is sech^2.
d/ds [ S * tanh( (s-sy)/w ) ] = S/w * sech^2( (s-sy)/w ).
This adds a "bump" in the slope, not a step.
To get a step in slope (yield), we need a function that goes from 0 to a constant.
tanh goes from -1 to 1.
So (1 + tanh(...))/2 goes from 0 to 1.
Let H(s) = 0.5 * (1 + tanh( (s - sy) / w )).
Then sigma = E * s + S * H(s).
Slope is E + S/w * sech^2(...).
This creates a smooth transition from slope E to slope E + S/w.
This is a very good model for yield.

Let's use this H(s) form.
sigma = (E0 + E1*t) * s + (S0 + S1*t) * 0.5 * (1 + tanh( (s - (sy0 + sy1*t)) / w )) + C

Parameters:
p0: E0
p1: E1
p2: sy0
p3: sy1
p4: w (width)
p5: S0
p6: S1
p7: C (offset)
p8: Quadratic term in s (E2 * s^2) to capture slight curvature in elastic region.
p9: Quadratic term in s for plastic region? Or just leave it.
Maybe p8 and p9 can be used to refine the shape.
Let's try:
sigma = (p0 + p1*t) * s + p8 * s^2 + (p5 + p6*t) * 0.5 * (1 + tanh( (s - (p2 + p3*t)) / w )) + p7

This has 8 parameters. We have 2 left.
We can add a second tanh for a later hardening stage or just leave them as 0 (optimizer will set them).
But unused parameters might confuse BFGS if they are exactly 0 and not changing the loss.
Better to give them a role.
How about a second yield point?
sigma = (p0 + p1*t) * s + p8 * s^2 
        + (p5 + p6*t) * H1(s) 
        + (p9) * H2(s)
Where H1 and H2 are two smooth step functions.
This allows for a more complex yield behavior (e.g., initial yield then hardening).

Let's stick to a robust 10-parameter model:
1. E0
2. E1
3. sy0
4. sy1
5. w1
6. S0
7. S1
8. C
9. E2 (quadratic elastic)
10. S2 (quadratic plastic hardening, scaled by H1)

sigma = (p0 + p1*t) * s + p8 * s^2 
        + (p5 + p6*t) * H1(s) 
        + p9 * s * H1(s)  # This adds a slope change in the plastic region proportional to s
        + p7

Wait, p9 * s * H1(s) means that after yield, the slope increases linearly with s?
d/ds [ p9 * s * H(s) ] = p9 * H(s) + p9 * s * H'(s).
This is a bit complex.
Simpler: Just use the 8 parameters and let the other 2 be small corrections or set to 0.
Actually, BFGS handles redundant parameters poorly if they are exactly zero.
Let's use all 10.

p0: E0
p1: E1
p2: sy0
p3: sy1
p4: w1
p5: S0
p6: S1
p7: C
p8: E2 (quadratic elastic)
p9: w2 (width of a second, broader transition for long-term hardening)

Model:
H1 = 0.5 * (1 + tanh( (s - (p2 + p3*t)) / p4 ))
H2 = 0.5 * (1 + tanh( (s - (p2 + p3*t)) / p9 ))  # Same center, different width?
Or different center? We don't have params for center2.
Let's just use one tanh and add quadratic terms.

sigma = (p0 + p1*t) * s + p8 * s^2 + (p5 + p6*t) * H1(s) + p7

This is 8 params.
Let's add:
p9: A coefficient for a linear term that is active only in plastic region?
sigma += p9 * max(0, s - sy) ? No, non-smooth.
sigma += p9 * s * H1(s) ?
Let's try that. It adds a term that grows with s after yield.

sigma = (p0 + p1*t) * s + p8 * s^2 + (p5 + p6*t) * H1(s) + p9 * s * H1(s) + p7

This looks reasonable.

Numerical stability:
Clip w to avoid division by zero.
Clip argument of tanh to avoid overflow (though tanh is stable, large args just go to 1).
"""
import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress prediction using a smooth hyperbolic tangent yield model.
    
    Parameters (10 total):
    p0: Elastic modulus E0
    p1: Temperature coefficient for E0
    p2: Yield strain center sy0
    p3: Temperature coefficient for sy0
    p4: Transition width w1
    p5: Yield amplitude S0
    p6: Temperature coefficient for S0
    p7: Constant offset C
    p8: Quadratic elastic coefficient E2
    p9: Plastic hardening slope coefficient (multiplied by s*H1)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    s = strain
    t = temp
    
    # 1. Elastic Terms
    # Linear and quadratic in strain, linear in temp
    E_eff = p0 + p1 * t
    elastic = E_eff * s + p8 * s * s
    
    # 2. Yield Transition Term
    # Smooth step function H(s) = 0.5 * (1 + tanh( (s - sy) / w ))
    # sy depends on temp
    sy = p2 + p3 * t
    
    # Clip width to prevent division by zero
    w = np.clip(p4, 1e-4, None)
    
    # Calculate argument for tanh
    # Clip to prevent extreme values, though tanh is stable
    arg = (s - sy) / w
    arg = np.clip(arg, -10, 10)
    
    H = 0.5 * (1.0 + np.tanh(arg))
    
    # Amplitude of the yield step
    S_eff = p5 + p6 * t
    
    # Yield offset
    yield_offset = S_eff * H
    
    # 3. Plastic Hardening Term
    # This term adds a slope that increases with strain in the plastic region
    # It is gated by H(s) so it is 0 in elastic region
    hardening = p9 * s * H
    
    # 4. Constant Offset
    offset = p7
    
    stress = elastic + yield_offset + hardening + offset
    
    return stress
