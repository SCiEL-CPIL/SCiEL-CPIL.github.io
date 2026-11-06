import numpy as np
import time

def integrate(f, t_span, y0):
    start_time = time.time()
    deadline = start_time + 290
    
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Extract A and b assuming linearity: f(t, y) = A(t)y + b(t)
    # For a "fixed linear system", we assume A and b are constant.
    # We verify this by checking A(t0) and A(t1).
    
    def extract_ab(t):
        b_t = np.array(f(t, np.zeros(n)), dtype=float)
        A_t = np.zeros((n, n))
        for i in range(n):
            ei = np.zeros(n)
            ei[i] = 1.0
            fi = np.array(f(t, ei), dtype=float)
            A_t[:, i] = fi - b_t
        return A_t, b_t

    try:
        A0, b0 = extract_ab(t0)
        A1, b1 = extract_ab(t1)
        
        # Check if constant
        if np.allclose(A0, A1, atol=1e-12, rtol=1e-12) and np.allclose(b0, b1, atol=1e-12, rtol=1e-12):
            # Constant system: y' = A y + b
            # Solution: y(t) = expm((t-t0)A) y0 + integral_{t0}^t expm((t-s)A) b ds
            # Let h = t1 - t0
            h = t1 - t0
            Ah = h * A0
            
            # Compute expm(Ah) and Phi(Ah) = integral_0^1 expm(s Ah) ds * h
            # Note: integral_{t0}^{t0+h} expm((t0+h-s)A) b ds = expm(hA) integral_0^h expm(-u A) b du
            # Let u = h - (t0+h-s) -> no, simpler:
            # y(t) = e^{Ah} y0 + \int_0^h e^{(h-\tau)A} b d\tau
            # = e^{Ah} y0 + e^{Ah} \int_0^h e^{-\tau A} b d\tau
            # Let J = \int_0^h e^{-\tau A} b d\tau.
            # Alternatively, compute \Phi(Ah) such that \int_0^h e^{(h-\tau)A} b d\tau = \Phi(Ah) b ?
            # No, \int_0^h e^{(h-\tau)A} b d\tau = e^{Ah} \int_0^h e^{-\tau A} b d\tau.
            # If b is constant, this is (e^{Ah} - I) A^{-1} b.
            # We can compute E = expm(Ah) and S = (E - I) A^{-1}.
            # Then particular solution is S @ b.
            
            # Using Taylor series for expm(Ah):
            # E = sum_{k=0}^K (Ah)^k / k!
            # S = sum_{k=1}^K (Ah)^k / (k * k!)  (since (e^X - I) X^{-1} = sum_{k=1}^∞ X^k / (k k!))
            
            E = np.eye(n)
            S = np.zeros((n, n))
            term = np.eye(n)
            
            # Determine number of terms needed for convergence
            # For 15x15 matrices, Taylor converges very fast if ||Ah|| is not huge.
            # Even if stiff (||Ah|| large), we might need more terms.
            # Let's cap at 100 terms to be safe, which is negligible in time.
            
            for k in range(1, 101):
                term = term @ Ah / k
                E = E + term
                S = S + term / k
                
                # Check convergence
                if np.max(np.abs(term)) < 1e-16:
                    break
            
            y_final = E @ y0 + S @ b0
            return y_final
        else:
            # If not constant, we need a more robust solver.
            # Given the task description "fixed 15-dimensional stiff linear system",
            # it is highly likely constant. But to be safe, use a simple adaptive RK45
            # or just a high-step RK4 if it varies.
            # However, the score penalty for calls is high.
            # Let's assume the "fixed" implies constant coefficients.
            # If they are not constant, the problem is ill-posed for a 16-call strategy.
            # We will fall back to a moderate RK4 to avoid total failure.
            
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

    except Exception:
        # Fallback
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

