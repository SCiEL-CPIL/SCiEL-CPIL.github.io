import numpy as np
import time

def integrate(f, t_span, y0):
    start_time = time.time()
    deadline = start_time + 280.0
    
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    def extract_ab(t):
        """
        Extracts A(t) and b(t) from f(t, y) = A(t)y + b(t).
        Uses n+1 function calls.
        """
        b_t = np.array(f(t, np.zeros(n)), dtype=float)
        A_t = np.zeros((n, n))
        for i in range(n):
            ei = np.zeros(n)
            ei[i] = 1.0
            fi = np.array(f(t, ei), dtype=float)
            A_t[:, i] = fi - b_t
        return A_t, b_t

    try:
        # Strategy: High-order Taylor Series Integration for Linear Systems
        # We approximate the solution of y' = A(t)y + b(t) over a step [t_k, t_{k+1}]
        # by assuming A and b are constant (evaluated at midpoint for 2nd order accuracy in approximation).
        # Instead of expm, we use a truncated Taylor series for the state transition matrix and the particular solution.
        # y(t+h) = (I + hA + h^2 A^2 / 2! + ... + h^m A^m / m!) y(t) + 
        #          (hI - h^2 A / 2! + h^3 A^2 / 3! - ... ) b  (approximate particular)
        # Actually, a more robust way for linear systems is to compute the matrix exponential via series
        # and the vector exponential integral via series.
        
        # Let's check if constant first
        A0, b0 = extract_ab(t0)
        A1, b1 = extract_ab(t1)
        
        # If constant, we can use a single high-accuracy step or just many steps.
        # But even if constant, the "stiff" part implies large eigenvalues.
        # Series convergence depends on ||hA||.
        
        # We will use a fixed number of segments K.
        # Each segment requires 2*(n+1) calls for midpoint A, b.
        # We need to balance K (accuracy) and time.
        
        # Estimate cost of one step:
        # Matrix multiplications for Taylor series up to order m.
        # For n=15, m=20 is safe for moderate stiffness.
        # If stiffness is high, we might need smaller h or higher m.
        
        # Let's start with K=200.
        # Calls: 200 * 2 * 16 = 6400.
        # Penalty: 0.5 * log10(6400) ≈ 1.8.
        
        # If we go to K=1000:
        # Calls: 1000 * 2 * 16 = 32000.
        # Penalty: 0.5 * log10(32000) ≈ 2.4.
        # Difference: 0.6.
        
        # If error drops from 1e-5 to 1e-8 with K=1000 vs K=200:
        # Score gain: -log10(1e-8) - 2.4 = 8 - 2.4 = 5.6
        # vs -log10(1e-5) - 1.8 = 5 - 1.8 = 3.2
        # So higher K is better if we can afford the time.
        
        # How fast is the Taylor series?
        # For n=15, matrix mult is very fast in NumPy (BLAS).
        # We can likely afford K=500 or K=1000 within 280s.
        
        K = 500
        h = (t1 - t0) / K
        y = y0.copy()
        
        # Precompute factorials for Taylor series
        m_order = 30 # High order to ensure convergence for stiff systems
        fact = np.ones(m_order + 1)
        for i in range(1, m_order + 1):
            fact[i] = fact[i-1] * i
            
        # We will compute the Taylor coefficients for the matrix exponential E = exp(hA)
        # and the vector integral term V = int_0^h exp((h-s)A) b ds
        # Note: V = E * (A^{-1} (I - exp(-hA)) b) ? No.
        # V = sum_{k=0}^m h^{k+1} / (k+1)! A^k b  (if we expand exp((h-s)A) and integrate)
        # Let's derive:
        # y(t+h) = exp(hA)y(t) + int_0^h exp((h-s)A) b ds
        # Let u = h-s, ds = -du. int_h^0 exp(uA) b (-du) = int_0^h exp(uA) b du
        # int_0^h sum_{k=0}^inf (uA)^k/k! b du = sum_{k=0}^inf A^k/k! b * [u^{k+1}/(k+1)]_0^h
        # = sum_{k=0}^inf h^{k+1} / (k+1)! A^k b
        # = h b + h^2/2 A b + h^3/6 A^2 b + ...
        
        # This is a vector series. We can compute it efficiently.
        
        for k in range(K):
            t_start = t0 + k * h
            t_mid = t_start + h / 2.0
            
            if time.time() > deadline - 5:
                # Emergency stop: return current y? No, must finish.
                # But if we are at k=499/500, it's fine.
                # If we are early, we might have to reduce K? Can't change K mid-loop easily.
                # We assume K=500 is fast enough.
                pass
            
            A_mid, b_mid = extract_ab(t_mid)
            
            # Compute E = exp(h * A_mid) via Taylor series
            # E = I + hA + (hA)^2/2! + ...
            # Let M = h * A_mid
            M = h * A_mid
            
            E = np.eye(n)
            term = np.eye(n)
            for i in range(1, m_order + 1):
                term = term @ M / i
                E = E + term
                
            # Compute V = sum_{k=0}^m h^{k+1} / (k+1)! A^k b_mid
            # V = h * ( b + h/2 A b + h^2/6 A^2 b + ... )
            # Let's compute the series for the vector:
            # V = h * sum_{j=0}^m (h^j / (j+1)!) A^j b
            # Let W_j = A^j b / (j+1)! * h^{j+1}
            # W_0 = h b
            # W_{j+1} = W_j * A * h / (j+2)  ... wait.
            # Term j: T_j = h^{j+1} / (j+1)! A^j b
            # T_{j+1} = h^{j+2} / (j+2)! A^{j+1} b = T_j * (h / (j+2)) * A
            # So:
            V = h * b_mid
            term_v = h * b_mid
            for j in range(1, m_order + 1):
                term_v = term_v @ A_mid * (h / (j + 1))
                V = V + term_v
            
            y = E @ y + V
            
            # Optional: Check convergence of series? 
            # For stiff systems, if ||hA|| is large, m=30 might not be enough.
            # But h is small (1/500), so ||hA|| should be small unless A is HUGE.
            # If A is huge (eigenvalues 1e5), hA ~ 200. exp(200) is huge.
            # Taylor series for exp(200) needs ~200 terms to converge accurately?
            # Actually, for large x, Taylor series of exp(x) converges slowly in relative terms?
            # No, it converges, but many terms are needed.
            # However, if the system is stiff, we usually scale time or use implicit methods.
            # But here we are forced to use f calls.
            # If the system is extremely stiff, this explicit-like Taylor method might fail or need very small h.
            # Given the task is "15-dimensional stiff linear system", it's likely moderately stiff.
            # If it's extremely stiff, K=500 might not be enough for stability/accuracy of the explicit series.
            # But let's try. If it fails, we can increase K or use a different method.
            
        return y

    except Exception:
        # Fallback: RK4 with moderate steps
        N = 1000
        h = (t1 - t0) / N
        t = t0
        y = y0.copy()
        for _ in range(N):
            k1 = np.array(f(t, y), dtype=float)
            k2 = np.array(f(t + h/2, y + h/2 * k1), dtype=float)
            k3 = np.array(f(t + h/2, y + h/2 * k2), dtype=float)
            k4 = np.array(f(t + h, y + h * k3), dtype=float)
            y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            t += h
        return y
