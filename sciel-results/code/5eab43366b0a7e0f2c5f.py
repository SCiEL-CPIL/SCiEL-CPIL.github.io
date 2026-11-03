# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def construct():
    start_time = time.time()
    deadline = 200.0  # Safety margin before 210s kill

    # Precompute constants for analytic integration
    # I_k = int x^(2k) exp(-x^2) dx from -inf to inf = 0.5 * Gamma(k + 0.5)
    # For P degree 12, P^2 degree 24 -> max k=12
    # For P', degree 11, (P')^2 degree 22 -> max k=11
    I_k = np.zeros(13)
    I_k[0] = np.sqrt(np.pi) / 2.0
    for k in range(1, 13):
        I_k[k] = I_k[k-1] * (k - 0.5)

    # Probabilists' Hermite polynomials in monomial form
    # Coefficients for x^0, x^2, x^4, x^6, x^8, x^10, x^12
    He0_coeffs = np.zeros(7)
    He0_coeffs[0] = 1.0

    He4_coeffs = np.zeros(7)
    He4_coeffs[0] = 3.0
    He4_coeffs[1] = -6.0
    He4_coeffs[2] = 1.0

    He8_coeffs = np.zeros(7)
    He8_coeffs[0] = 105.0
    He8_coeffs[1] = -420.0
    He8_coeffs[2] = 210.0
    He8_coeffs[3] = -28.0
    He8_coeffs[4] = 1.0

    He12_coeffs = np.zeros(7)
    He12_coeffs[0] = 9240.0
    He12_coeffs[1] = -12096.0
    He12_coeffs[2] = 30240.0
    He12_coeffs[3] = -15120.0
    He12_coeffs[4] = 1605.0
    He12_coeffs[5] = -42.0
    He12_coeffs[6] = 1.0

    def get_P_coeffs(c0, c4, c8):
        # P(0) = c0 + 3*c4 + 105*c8 + 9240*c12 = 0
        c12 = -(c0 + 3.0 * c4 + 105.0 * c8) / 9240.0
        P_coeffs = c0 * He0_coeffs + c4 * He4_coeffs + c8 * He8_coeffs + c12 * He12_coeffs
        return P_coeffs

    def compute_integrals(a):
        """
        Compute I0, I1, I2 from monomial coefficients a of P(x)
        P(x) = sum a_j x^(2j)
        I0 = int P^2 exp(-x^2) dx
        I1 = int x^2 P^2 exp(-x^2) dx
        I2 = int (P')^2 exp(-x^2) dx
        """
        I0 = 0.0
        for i in range(7):
            if a[i] == 0: continue
            for j in range(i, 7):
                if a[j] == 0: continue
                term = a[i] * a[j] * I_k[i+j]
                if i != j:
                    term *= 2.0
                I0 += term

        I1 = 0.0
        for i in range(7):
            if a[i] == 0: continue
            for j in range(i, 7):
                if a[j] == 0: continue
                term = a[i] * a[j] * I_k[i+j+1]
                if i != j:
                    term *= 2.0
                I1 += term

        I2 = 0.0
        for i in range(1, 7):
            if a[i] == 0: continue
            bi = 2.0 * i * a[i]
            for j in range(i, 7):
                if a[j] == 0: continue
                bj = 2.0 * j * a[j]
                term = bi * bj * I_k[i+j-1]
                if i != j:
                    term *= 2.0
                I2 += term

        return I0, I1, I2

    def objective(params):
        if time.time() > deadline:
            return 1e10
        c0, c4, c8 = params
        a = get_P_coeffs(c0, c4, c8)
        I0, I1, I2 = compute_integrals(a)
        if I0 < 1e-12:
            return 1e10
        return (I1 * I2) / (I0 * I0)

    def analytic_gradient(params):
        """
        Compute analytic gradient of objective = (I1*I2)/(I0^2)
        d/dx [(I1*I2)/(I0^2)] = (I2*I1' - I1*I2')/I0^2 - 2*I1*I2*I0'/I0^3
        = (I2*I1' - I1*I2')/I0^2 - 2*(I1*I2*I0')/I0^3
        """
        c0, c4, c8 = params
        a = get_P_coeffs(c0, c4, c8)
        I0, I1, I2 = compute_integrals(a)
        if I0 < 1e-12:
            return np.array([1e10, 1e10, 1e10])
        
        # Compute derivatives of I0, I1, I2 with respect to c0, c4, c8
        # a = c0*He0 + c4*He4 + c8*He8 + c12*He12
        # da/dc0 = He0, da/dc4 = He4, da/dc8 = He8
        # da/dc12 = He12, dc12/dc0 = -1/9240, dc12/dc4 = -3/9240, dc12/dc8 = -105/9240
        
        da_dc0 = He0_coeffs - He12_coeffs / 9240.0
        da_dc4 = He4_coeffs - 3.0 * He12_coeffs / 9240.0
        da_dc8 = He8_coeffs - 105.0 * He12_coeffs / 9240.0
        
        grad = np.zeros(3)
        
        for idx, da in enumerate([da_dc0, da_dc4, da_dc8]):
            # dI0/dc = 2 * sum_i a_i * I_k[i+j] * da_j (for all i,j)
            # More precisely: I0 = sum_i sum_j a_i a_j I_k[i+j]
            # dI0/dx = 2 * sum_i a_i (da/dc)_i * I_k[i+j] summed over j
            dI0 = 0.0
            dI1 = 0.0
            dI2 = 0.0
            
            for i in range(7):
                if da[i] == 0: continue
                for j in range(7):
                    if a[j] == 0: continue
                    dI0 += 2.0 * a[i] * da[j] * I_k[i+j]
                    dI1 += 2.0 * a[i] * da[j] * I_k[i+j+1]
                    # For I2: I2 = sum_i sum_j (2i*a_i)(2j*a_j) I_k[i+j-1] for i,j>=1
                    if i >= 1 and j >= 1:
                        bi = 2.0 * i * a[i]
                        bj = 2.0 * j * a[j]
                        # d/dx of bi*bj = (2i*da_i)*bj + bi*(2j*da_j)
                        dI2 += (2.0 * i * da[i]) * bj * I_k[i+j-1] + bi * (2.0 * j * da[j]) * I_k[i+j-1]
            
            # d/dx [(I1*I2)/(I0^2)] = (I2*dI1 + I1*dI2)/I0^2 - 2*I1*I2*dI0/I0^3
            grad[idx] = (I2 * dI1 + I1 * dI2) / (I0 * I0) - 2.0 * I1 * I2 * dI0 / (I0 * I0 * I0)
        
        return grad

    # Seed values from the initial program
    c0_init = -113.0 / 100.0
    c4_init = 1.0 / 25.0
    c8_init = 1.0 / 3240.0
    seed = np.array([c0_init, c4_init, c8_init])
    
    best_score = objective(seed)
    best_coeffs = seed.copy()

    # Define bounds for Differential Evolution
    bounds = [(-2.0, 2.0), (-1.0, 1.0), (-0.1, 0.1)]
    
    # Run Differential Evolution with multiple seeds
    try:
        if time.time() < deadline - 15:
            result = differential_evolution(
                objective, 
                bounds, 
                seed=42, 
                maxiter=800, 
                popsize=30, 
                tol=1e-12, 
                polish=False,
                init='sobol'
            )
            if result.fun < best_score:
                best_score = result.fun
                best_coeffs = result.x.copy()
    except Exception:
        pass

    # Multi-start with analytic gradient: try many starting points
    if time.time() < deadline - 10:
        # Generate diverse starting points
        rng = np.random.RandomState(123)
        start_points = [seed.copy()]
        
        # Add points from a grid
        for c0 in np.linspace(-1.5, 1.5, 5):
            for c4 in np.linspace(-0.5, 0.5, 4):
                for c8 in np.linspace(-0.05, 0.05, 3):
                    start_points.append(np.array([c0, c4, c8]))
        
        # Add random points
        for _ in range(20):
            start_points.append(rng.uniform([-2.0, -1.0, -0.1], [2.0, 1.0, 0.1]))
        
        for sp in start_points:
            if time.time() > deadline - 5:
                break
            try:
                res = minimize(
                    objective,
                    sp,
                    jac=analytic_gradient,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-18, 'gtol': 1e-14}
                )
                if res.fun < best_score:
                    best_score = res.fun
                    best_coeffs = res.x.copy()
            except Exception:
                pass

    # Final high-precision polish
    if time.time() < deadline - 3:
        try:
            res = minimize(
                objective,
                best_coeffs,
                jac=analytic_gradient,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-20, 'gtol': 1e-16}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass

    return best_coeffs

# EVOLVE-BLOCK-END
