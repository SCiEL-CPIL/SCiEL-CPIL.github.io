import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.special import roots_legendre
import time

def construct():
    t0 = time.time()
    deadline = 85.0  # Safety margin before 90s kill

    # 1. Define the physics
    def E_of_z(z, w0, wa):
        """
        E(z) = H(z)/H0 for CPL model in flat universe.
        w(z) = w0 + wa * z
        E(z) = (1+z)^{3/2} * (1+z)^{3*(w0+wa)} * (1+z)^{-3*wa}
             = (1+z)^{3/2 + 3*w0 + 3*wa - 3*wa} ? No.
        
        Standard formula:
        E(z) = (1+z)^{3/2} (1+z)^{3(w_0+w_a)} (1+z)^{-3w_a}
             = (1+z)^{3/2 + 3w_0}
        Wait, the standard CPL parameterization is:
        E(z) = (1+z)^{3/2} (1+z)^{3(w_0+w_a)} (1+z)^{-3w_a}
        Let's check the exponent:
        3/2 + 3(w0+wa) - 3wa = 3/2 + 3w0 + 3wa - 3wa = 3/2 + 3w0.
        This implies E(z) is independent of wa? That's wrong.
        
        Correct CPL formula:
        E(z) = (1+z)^{3/2} (1+z)^{3(w_0+w_a)} (1+z)^{-3w_a}
        Actually, the term is usually written as:
        E(z) = (1+z)^{3/2} (1+z)^{3w_0 + 3w_a} (1+z)^{-3w_a} ?
        
        Let's derive from scratch.
        H(z) = H0 (1+z)^{3/2} (1+z)^{3(w_0+w_a)} (1+z)^{-3w_a}
        Exponent of (1+z): 3/2 + 3w_0 + 3w_a - 3w_a = 3/2 + 3w_0.
        This is the "Chevallier-Polarski-Linder" parametrization?
        No, the standard CPL is:
        w(z) = w_0 + w_a (1 - a) = w_0 + w_a z
        Then \int_0^z (1+z')^{-1} E(z') dz' ...
        
        The luminosity distance is:
        d_L = \frac{c}{H_0} (1+z) \int_0^z \frac{dz'}{E(z')}
        
        The integrand in the task is:
        d_L/(c/H0) = (1+z) z \sum ... 
        Wait, the task says:
        dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i)
        This implies the integral is approximated by the sum.
        The integral is \int_0^z \frac{dt}{E(t)}?
        No, the standard formula is \int_0^z \frac{dz'}{E(z')}.
        The task formula has an extra factor of z?
        "dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i)"
        This looks like it's approximating \int_0^z \frac{t}{E(t)} dt? Or \int_0^z \frac{1}{E(t)} dt?
        If the formula is (1+z) * [Sum w_i / E(a_i)], and the nodes are in [0,1], 
        usually the quadrature approximates \int_0^1 f(t) dt.
        Here the upper limit is z. 
        But the nodes are fixed in [0,1]. 
        The formula given is: (1+z)*z*sum_i weights[i]/E(a_i).
        This is strange. Usually, for a fixed quadrature rule on [0,1], you approximate \int_0^1 f(t) dt.
        If the integral is \int_0^z \frac{dt}{E(t)}, you can't use a fixed rule on [0,1] directly unless you change variables or the rule is for the specific z?
        
        Re-reading the task carefully:
        "dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i), where a_i=1/(1+z*nodes[i])"
        This implies the integral being approximated is \int_0^1 \frac{t}{E(a(t))} dt?
        Let's check the change of variable.
        Let $t = z' / z$. Then $z' = z t$, $dz' = z dt$.
        \int_0^z \frac{dz'}{E(z')} = \int_0^1 \frac{z dt}{E(zt)} = z \int_0^1 \frac{dt}{E(zt)}$.
        Then $d_L = \frac{c}{H_0} (1+z) z \int_0^1 \frac{dt}{E(zt)}$.
        So the integrand to be quadratured is $g(t) = \frac{1}{E(zt)}$.
        The nodes are $t_i \in [0,1]$.
        The values are evaluated at $a_i = 1/(1+z t_i)$.
        So $E(a_i) = E(z_i)$ where $z_i = z t_i$.
        
        So the task is: Find nodes $t_i$ and weights $w_i$ for $t \in [0,1]$ such that
        $\sum w_i \frac{1}{E(z t_i)} \approx \int_0^1 \frac{dt}{E(z t)}$
        for all $z$ and all models.
        
        This is a **universal** quadrature for the family of functions $f_z(t) = \frac{1}{E(z t)}$.
        
        This makes sense. The nodes are in [0,1] (the t-space).
        
        Okay, so the integrand depends on z.
        $f_z(t) = \frac{1}{E(z t)}$.
        $E(z t) = (1+z t)^{3/2 + 3w_0 + 3w_a z t - 3w_a}$.
        Wait, the exponent in E(z) for CPL is:
        $E(z) = (1+z)^{3/2} (1+z)^{3(w_0+w_a)} (1+z)^{-3w_a}$.
        Let's verify the exponent again.
        $w(z) = w_0 + w_a z$.
        $\ln E(z) = \frac{3}{2} \ln(1+z) + 3 \int_0^z (1+w(z')) \frac{dz'}{1+z'}$.
        $\int_0^z (1+w_0+w_a z') \frac{dz'}{1+z'} = (1+w_0) \ln(1+z) + w_a \int_0^z \frac{z'}{1+z'} dz'$.
        $\int_0^z \frac{z'}{1+z'} dz' = \int_0^z (1 - \frac{1}{1+z'}) dz' = z - \ln(1+z)$.
        So $\ln E(z) = \frac{3}{2} \ln(1+z) + 3(1+w_0) \ln(1+z) + 3w_a (z - \ln(1+z))$.
        $\ln E(z) = (3/2 + 3 + 3w_0 - 3w_a) \ln(1+z) + 3w_a z$.
        $\ln E(z) = (9/2 + 3w_0 - 3w_a) \ln(1+z) + 3w_a z$.
        $E(z) = (1+z)^{9/2 + 3w_0 - 3w_a} e^{3w_a z}$.
        
        This is the correct CPL E(z).
        
        So $f_z(t) = \frac{1}{E(z t)} = (1+z t)^{-(9/2 + 3w_0 - 3w_a)} e^{-3w_a z t}$.
        
        We need to approximate $\int_0^1 f_z(t) dt$ with $\sum w_i f_z(t_i)$.
        
        This is a much clearer problem.
        The function $f_z(t)$ is smooth in $t \in [0,1]$.
        It depends on $z$ and the model parameters.
        
        Strategy:
        1. Sample a grid of $(z, w_0, w_a)$.
        2. For each sample, compute the reference integral using high-precision `quad`.
        3. Optimize nodes $t_i$ and weights $w_i$ to minimize the worst-case relative error over all samples.
        
        Initial guess: Gauss-Legendre on [0,1].
        
        This is a standard problem. The previous program did this but used DE.
        We will use local optimization from GL start.
        """
        # Correct E(z) implementation
        def E_z(z, w0, wa):
            # E(z) = (1+z)^(9/2 + 3w0 - 3wa) * exp(3wa*z)
            # Note: for large z, exp term dominates.
            # To avoid overflow, we might need care, but z is in [0,1] for the quadrature variable t, 
            # and z (redshift) in validation might be larger?
            # The task says "disjoint public validation cosmologies". 
            # Usually z is up to 1 or 2.
            # The quadrature is on t in [0,1]. The argument is z*t.
            # If z is large, z*t can be large.
            # But E(z) grows fast, so 1/E(z) decays fast.
            
            # Let's stick to the formula.
            base = 1.0 + z
            exponent = 4.5 + 3.0*w0 - 3.0*wa
            power_term = base**exponent
            exp_term = np.exp(3.0*wa*z)
            return power_term * exp_term

        def integrand_t(t, z, w0, wa):
            return 1.0 / E_z(z*t, w0, wa)

        # Reference integral
        def ref_integral(z, w0, wa):
            # Integrate from 0 to 1
            val, _ = quad(integrand_t, 0, 1, args=(z, w0, wa), limit=100)
            return val

        # Training set
        # z values: 0.1, 0.5, 1.0, 2.0
        z_vals = [0.1, 0.5, 1.0, 2.0]
        w0_vals = [-1.5, -1.2, -1.0, -0.8, -0.5, 0.0, 0.5]
        wa_vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
        
        train_data = []
        for z in z_vals:
            for w0 in w0_vals:
                for wa in wa_vals:
                    # Skip extreme cases that might cause numerical issues in quad?
                    # E(z) should be positive.
                    ref = ref_integral(z, w0, wa)
                    if ref > 1e-10:
                        train_data.append((z, w0, wa, ref))
        
        n_train = len(train_data)
        
        def compute_errors(nodes, weights):
            errors = np.zeros(n_train)
            for i, (z, w0, wa, ref) in enumerate(train_data):
                # Evaluate quadrature
                # f(t) = 1/E(z*t)
                # nodes are t_i
                z_t = z * nodes
                E_vals = E_z(z_t, w0, wa)
                f_vals = 1.0 / E_vals
                q_val = np.dot(weights, f_vals)
                errors[i] = abs(q_val - ref) / abs(ref)
            return errors

        def objective(params):
            nodes = params[:8]
            weights = params[8:15] # 7 weights
            # 8th weight
            w8 = 1.0 - np.sum(weights)
            if w8 < 1e-12:
                return 1e10
            all_weights = np.append(weights, w8)
            if np.any(all_weights < 1e-12):
                return 1e10
            if np.any(nodes < 1e-6) or np.any(nodes > 1 - 1e-6):
                return 1e10
            # Check ordering
            if not np.all(np.diff(nodes) > 1e-6):
                return 1e10
                
            errors = compute_errors(nodes, all_weights)
            return np.max(errors)

        # Initial guess: Gauss-Legendre
        x, w = roots_legendre(8)
        # Map from [-1, 1] to [0, 1]
        nodes_init = 0.5 * (x + 1.0)
        weights_init = 0.5 * w
        
        # Parameters for optimizer
        # Nodes: 8
        # Weights: 7 (8th is derived)
        params_init = np.concatenate([nodes_init, weights_init[:7]])
        
        bounds = [(1e-5, 1-1e-5) for _ in range(8)] + [(1e-8, 1.0) for _ in range(7)]
        
        # Optimize
        best_params = params_init
        best_err = objective(params_init)
        
        # Powell method with bounds
        try:
            res = minimize(objective, params_init, method='Powell', bounds=bounds, 
                           options={'maxiter': 200, 'ftol': 1e-12, 'disp': False})
            if res.fun < best_err:
                best_err = res.fun
                best_params = res.x
        except Exception:
            pass
            
        # Nelder-Mead if time remains
        if time.time() - t0 < deadline - 10:
            try:
                # NM doesn't support bounds directly, but we can use it on the interior
                # Or just use the result from Powell
                pass
            except Exception:
                pass

        # Extract results
        nodes = best_params[:8]
        weights_7 = best_params[8:15]
        w8 = 1.0 - np.sum(weights_7)
        weights = np.append(weights_7, w8)
        
        # Final safety checks
        if np.any(weights < 0) or np.any(nodes <= 0) or np.any(nodes >= 1):
            # Fallback to GL
            nodes = nodes_init
            weights = weights_init
        else:
            # Ensure sum is exactly 1
            weights = weights / np.sum(weights)
            # Ensure strict ordering (should be fine)
            
        return {"nodes": nodes, "weights": weights}
