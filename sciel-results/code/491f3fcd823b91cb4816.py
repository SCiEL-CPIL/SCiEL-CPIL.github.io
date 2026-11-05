import numpy as np
import scipy.linalg as la

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Check if the system is time-independent
    # Evaluate f at t0 and mid-point with same y
    f_t0 = f(t0, y)
    f_mid = f((t0 + t1) / 2, y)
    
    is_constant = np.allclose(f_t0, f_mid, rtol=1e-10, atol=1e-10)
    
    if is_constant:
        # Extract A and b at t0
        b = f(t0, np.zeros(n))
        A = np.zeros((n, n))
        for j in range(n):
            e_j = np.zeros(n)
            e_j[j] = 1.0
            A[:, j] = f(t0, e_j) - b
        
        # Build augmented matrix for exact solution of y' = Ay + b
        # State vector is [y; 1]
        # d/dt [y; 1] = [Ay + b; 0] = [[A, b], [0, 0]] [y; 1] 
        # Wait, the augmented trick for constant b uses:
        # d/dt [y; c] = [Ay + b*c; 0] with c(0)=0? No.
        # Let's use the standard trick:
        # y(t) = e^{At}y(0) + \int_0^t e^{A(t-\tau)}b d\tau
        # Let M = [[A, b], [0, 1]]
        # e^M = [[e^A, \int_0^1 e^{A(1-s)}b ds], [0, 1]]
        # Then [y(1); 1] = e^M [y(0); 0]
        # y(1) = e^A y(0) + (\int_0^1 e^{A(1-s)}b ds) * 0 ? No.
        # The vector multiplied is [y(0); 0].
        # So y(1) = (e^A y(0)) + (integral block * 0) = e^A y(0). This is wrong.
        # The integral term must be added.
        # Let's look at the block multiplication:
        # [[e^A, I_int], [0, 1]] * [y0; 0] = [e^A y0; 0].
        # This gives y(1) = e^A y0. It misses the forcing term.
        # Why? Because the forcing term is constant b, not b*c.
        # If we define the system as y' = Ay + b, we can homogenize it by adding a component z=1.
        # y' = Ay + b*1
        # z' = 0
        # So the matrix is [[A, b], [0, 0]].
        # e^M = [[e^A, \int_0^1 e^{A(1-s)} b ds], [0, 0]]? No, the (2,2) element is e^0=1.
        # Let M = [[A, b], [0, 0]].
        # e^M = [[e^A, \int_0^1 e^{A(1-s)} b ds], [0, 1]]?
        # Let's check derivative of e^{Mt}.
        # d/dt (e^{Mt})_{12} = (M e^{Mt})_{12} = A (e^{Mt})_{12} + b (e^{Mt})_{22}.
        # (e^{Mt})_{22} = e^0 t? No, M_{22}=0, so e^{Mt}_{22}=1.
        # So d/dt I(t) = A I(t) + b.
        # I(0) = 0.
        # Solution I(t) = \int_0^t e^{A(t-s)} b ds.
        # So e^M_{12} = \int_0^1 e^{A(1-s)} b ds.
        # Then [y(1); 1] = e^M [y(0); 1] = [e^A y0 + I(1)*1; 1].
        # Yes! We must multiply by [y0; 1].
        
        M_aug = np.zeros((n + 1, n + 1))
        M_aug[:n, :n] = A
        M_aug[:n, n] = b
        M_aug[n, n] = 0.0
        
        E_aug = la.expm(M_aug)
        
        y_aug_init = np.zeros(n + 1)
        y_aug_init[:n] = y
        y_aug_init[n] = 1.0
        
        y_end_aug = E_aug @ y_aug_init
        y_end = y_end_aug[:n]
        
        return y_end
        
    else:
        # Time-dependent system. Use piecewise constant approximation.
        # Choose number of steps. 20 is a good balance.
        steps = 20
        h = (t1 - t0) / steps
        t = t0
        
        for _ in range(steps):
            # Extract A and b at current t
            b = f(t, np.zeros(n))
            A = np.zeros((n, n))
            for j in range(n):
                e_j = np.zeros(n)
                e_j[j] = 1.0
                A[:, j] = f(t, e_j) - b
            
            # Build augmented matrix for this step
            # We need to solve y' = Ay + b on [t, t+h]
            # Using the same trick, but scaled by h?
            # The matrix for the step is M_step = [[A*h, b*h], [0, 0]]?
            # No, the system is y' = Ay + b.
            # Let's use the exact solution for constant A, b over interval h.
            # y(t+h) = e^{Ah} y(t) + \int_t^{t+h} e^{A(t+h-\tau)} b d\tau
            # Let s = t+h-\tau, ds = -d\tau.
            # Integral = \int_0^h e^{As} b ds.
            # Using augmented matrix M = [[A, b], [0, 0]], e^{M h} = [[e^{Ah}, \int_0^h e^{A(h-s)} b ds], [0, 1]].
            # Wait, the integral in the block is \int_0^h e^{A(h-s)} b ds?
            # d/dt I(t) = A I(t) + b, I(0)=0.
            # I(h) = \int_0^h e^{A(h-s)} b ds.
            # Yes.
            # So we compute E_step = expm([[A, b], [0, 0]] * h)?
            # No, expm(M * h) where M = [[A, b], [0, 0]].
            # M*h = [[A*h, b*h], [0, 0]].
            # Let's verify.
            # d/dt y = A y + b.
            # Let z = [y; 1].
            # z' = [[A, b], [0, 0]] z.
            # Solution z(t+h) = exp([[A, b], [0, 0]] h) z(t).
            
            M_step = np.zeros((n + 1, n + 1))
            M_step[:n, :n] = A * h
            M_step[:n, n] = b * h
            M_step[n, n] = 0.0
            
            E_step = la.expm(M_step)
            
            y_aug = np.zeros(n + 1)
            y_aug[:n] = y
            y_aug[n] = 1.0
            
            y_aug = E_step @ y_aug
            y = y_aug[:n]
            
            t += h
            
        return y
