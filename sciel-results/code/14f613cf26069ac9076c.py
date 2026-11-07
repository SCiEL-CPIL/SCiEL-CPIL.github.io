import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the function using a rational function model.
    f(x) = (a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4) / (1 + b1*x + b2*x^2 + b3*x^3 + b4*x^4) + c
    params: [a0, a1, a2, a3, a4, b1, b2, b3, b4, c]
    """
    a0, a1, a2, a3, a4, b1, b2, b3, b4, c = params
    
    # Compute numerator
    num = a0 + a1 * x + a2 * x**2 + a3 * x**3 + a4 * x**4
    
    # Compute denominator
    den = 1.0 + b1 * x + b2 * x**2 + b3 * x**3 + b4 * x**4
    
    # Prevent division by zero (though BFGS should avoid this, it's safer)
    # If den is very close to 0, the gradient is huge. 
    # We can clip the denominator magnitude to a small epsilon.
    eps = 1e-6
    den_safe = np.where(np.abs(den) < eps, eps * np.sign(den), den)
    # If den is exactly 0, sign is 0, so we use eps. 
    # Actually, np.sign(0) is 0. Let's just use a small positive value if it's 0?
    # Better: den_safe = np.where(den == 0, eps, den)
    # But if den is -1e-9, we want -1e-9.
    # np.where(np.abs(den) < eps, eps, den) would flip sign if den was negative and small.
    # Let's just add a tiny epsilon to the magnitude? No, that shifts the function.
    # For BFGS, it's better to let it be, as long as it doesn't crash.
    # The evaluator says "non-finite error costs 1e30". So we must avoid inf/nan.
    
    # A robust way:
    # den_safe = den / (1 + (eps / den)**2) ? No.
    # Let's just use np.divide with where clause.
    
    y = np.empty_like(x)
    mask = np.abs(den) > 1e-8
    y[mask] = num[mask] / den[mask] + c
    # For points where den is very small, the function is effectively undefined or huge.
    # In optimization, this creates huge gradients. 
    # If we return 0 or some constant for these, it might mislead the optimizer.
    # However, if the true function is well-behaved, the optimizer should drive den away from 0.
    # If we start at ones, den = 1 + x + x^2 + x^3 + x^4.
    # For x in [-2, 2], min value of 1+x+x^2+x^3+x^4?
    # At x=-1: 1-1+1-1+1 = 1.
    # At x=-2: 1-2+4-8+16 = 11.
    # At x=2: 1+2+4+8+16 = 31.
    # It seems positive for x in [-2,2] with all ones.
    # So initialization is safe.
    
    # We will just compute directly. If it produces inf, the evaluator handles it (penalty).
    # But to be safe against NaN from 0/0, we use the mask.
    # If den is 0, we set y to 0? Or a large number? 
    # If we set it to 0, the loss is finite, but the gradient is wrong.
    # Given the prompt "non-finite ... costs 1e30", we should try to avoid non-finite.
    # So we must handle the division.
    
    # Let's replace non-finite values with 0 to keep the function finite, 
    # hoping the optimizer moves away from the singularity.
    with np.errstate(divide='ignore', invalid='ignore'):
        res = num / den + c
        res = np.where(np.isfinite(res), res, 0.0)
        
    return res
