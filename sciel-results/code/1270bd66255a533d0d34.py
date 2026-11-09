"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp

def construct():
    # Generate reference data using numerical ODE integration
    # to fit the 6 coefficients c0..c5
    
    # Test points for w0, wa, and a
    # We assume standard Omega_m0 = 0.3
    Om0 = 0.3
    OmDE0 = 1.0 - Om0
    
    w0_vals = [-1.0, -0.8, -0.5, 0.0]
    wa_vals = [-0.5, 0.0, 0.5]
    a_vals = np.linspace(0.2, 1.0, 10)
    
    X = []
    y = []
    
    for w0 in w0_vals:
        for wa in wa_vals:
            for a in a_vals:
                # Calculate Omega_m(a) and H(a) for the ODE
                # We need to integrate from a=1 to a=a_target
                # Initial conditions at a=1: D(1)=1, f(1) approx 0.55
                # f = a * D' / D => D'(1) = f(1) * D(1) / a(1) = 0.55
                
                def ode_func(a, y):
                    D, Dp = y
                    # w(a)
                    w = w0 + wa * (1.0 - a)
                    # Omega_m(a)
                    Om_a = Om0 * a**-3.0
                    # Omega_DE(a)
                    # Integral of a^-3(1+w)
                    # For CPL: Omega_DE(a) = Omega_DE0 * a^-3(1+w0+wa) * exp(3*wa*(1-a))
                    OmDE_a = OmDE0 * a**(-3.0*(1.0+w0+wa)) * np.exp(3.0*wa*(1.0-a))
                    Om_tot = Om_a + OmDE_a
                    
                    # H^2(a) proportional to Om_tot
                    # H(a)/H(1) = sqrt(Om_tot / Om_tot(1))
                    Om_tot_1 = Om0 + OmDE0
                    H_ratio = np.sqrt(Om_tot / Om_tot_1)
                    
                    # d ln H / d ln a
                    # H^2 ~ a^-3 + a^-3(1+w) ...
                    # d ln H / d ln a = 0.5 * (d ln H^2 / d ln a)
                    # H^2 = H0^2 [ Om0 a^-3 + OmDE0 a^-3(1+w0+wa) e^{3wa(1-a)} ]
                    # Let term1 = Om0 a^-3, term2 = ...
                    # d ln H^2 / d ln a = (a/H^2) d(H^2)/da
                    # d(H^2)/da = H0^2 [ -3 Om0 a^-4 + d(OmDE term)/da ]
                    # This is getting complex. Let's use the analytic derivative of ln H.
                    # ln H = 0.5 ln( Om0 a^-3 + OmDE0 a^-3(1+w0+wa) e^{3wa(1-a)} ) - const
                    # Let's just compute H and its derivative numerically or analytically.
                    # Analytic:
                    # d ln H / d ln a = 0.5 * ( a * d(H^2)/da ) / H^2
                    # d(H^2)/da = H0^2 * ( -3*Om0*a^-4 + OmDE0 * d/da [ a^-3(1+w0+wa) e^{3wa(1-a)} ] )
                    # d/da [ a^p e^{q(1-a)} ] = p a^{p-1} e^{q(1-a)} - q a^p e^{q(1-a)}
                    # p = -3(1+w0+wa), q = -3wa
                    # term2_der = OmDE0 * e^{3wa(1-a)} * ( p*a^{-3(1+w0+wa)-1} - 3wa*a^{-3(1+w0+wa)} )
                    # This is tedious.
                    
                    # Alternative: Use the identity for CPL:
                    # d ln H / d ln a = -1.5 + 1.5 * OmDE(a) * (1+w(a)) / Om_tot(a) ? No.
                    # H^2 = H0^2 ( Om_m a^-3 + Om_DE a^-3(1+w) ... )
                    # Actually, for CPL, the effective equation of state is w(a).
                    # d ln H / d ln a = -1.5 (1 + w_eff) where w_eff is not simply w(a) for CPL?
                    # No, for CPL, the acceleration equation gives:
                    # q = 0.5 Om_m + 0.5 Om_DE (1+w)
                    # d ln H / d ln a = - (1 + q) = -1 - 0.5 Om_m - 0.5 Om_DE (1+w)
                    
                    q_cosmo = 0.5 * Om_a + 0.5 * OmDE_a * (1.0 + w)
                    dlnH_dlna = -1.0 - q_cosmo
                    
                    # ODE: D'' + (3 + dlnH_dlna) D' - 1.5 Om_m D = 0
                    # Let y = [D, D']
                    # y' = [D', D'']
                    # D'' = 1.5 Om_a D - (3 + dlnH_dlna) D'
                    
                    Dpp = 1.5 * Om_a * D - (3.0 + dlnH_dlna) * Dp
                    return [Dp, Dpp]
                
                # Integrate from a=1 to a
                # If a < 1, integrate backwards? solve_ivp handles direction if t_span is (1, a)
                try:
                    sol = solve_ivp(ode_func, [1.0, a], [1.0, 0.55], method='RK45', rtol=1e-8, atol=1e-10, dense_output=False)
                    if sol.success:
                        D_end = sol.y[0, -1]
                        Dp_end = sol.y[1, -1]
                        # f = a * D' / D
                        f_end = a * Dp_end / D_end
                        # gamma = d ln f / d ln a
                        # We need f at nearby points to get derivative?
                        # Or we can compute gamma from the ODE relation?
                        # f = Omega_m^gamma => ln f = gamma ln Om_m
                        # d ln f / d ln a = gamma
                        # We have f(a). We need f'(a) or f at a+da.
                        # Let's just use the fact that we can evaluate f at a and a+epsilon.
                        # But we are in a loop. It's better to solve for a range and use finite differences.
                        
                        # For the sake of speed in construct(), let's just use the value of f 
                        # and assume the local slope is well approximated by the ODE?
                        # Actually, gamma is the index. 
                        # We can estimate gamma by solving for f at a and a*1.01.
                        # This doubles the cost. 
                        # Alternative: Use the analytic expression for gamma in terms of f and Om?
                        # No, gamma is the unknown.
                        
                        # Let's just sample f at a and a+0.01.
                        a2 = a * 1.01 if a < 0.99 else a * 0.99
                        sol2 = solve_ivp(ode_func, [1.0, a2], [1.0, 0.55], method='RK45', rtol=1e-8, atol=1e-10)
                        if sol2.success:
                            D_end2 = sol2.y[0, -1]
                            Dp_end2 = sol2.y[1, -1]
                            f_end2 = a2 * Dp_end2 / D_end2
                            
                            # gamma = (ln f2 - ln f1) / (ln a2 - ln a1)
                            if f_end > 0 and f_end2 > 0:
                                gamma_ref = (np.log(f_end2) - np.log(f_end)) / (np.log(a2) - np.log(a))
                                # Construct row for X
                                q = 1.0 - Om_a
                                u = 1.0 + w0 + wa * (1.0 - a)
                                row = [1.0, q, q**2, u, wa * a, u * q]
                                X.append(row)
                                y.append(gamma_ref)
                except Exception:
                    pass
    
    if len(X) > 0:
        X_arr = np.array(X)
        y_arr = np.array(y)
        # Solve least squares
        coeffs, residuals, rank, s = np.linalg.lstsq(X_arr, y_arr, rcond=None)
        # Clip to [-10, 10]
        coeffs = np.clip(coeffs, -10.0, 10.0)
        return list(coeffs)
    else:
        # Fallback
        return [0.55, 0.0, 0.0, 0.18, 0.06, -0.08]

# EVOLVE-BLOCK-END
