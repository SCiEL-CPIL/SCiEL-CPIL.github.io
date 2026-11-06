    import numpy as np
    from scipy.sparse.linalg import LinearOperator, expm_multiply

    def integrate(f, t_span, y0):
        t0, t1 = t_span
        y0 = np.array(y0, dtype=float)
        n = len(y0)
        h = t1 - t0
        
        # Define the linear operator
        # f takes (t, y). We fix t=t0.
        def matvec(v):
            return f(t0, v)
            
        A_op = LinearOperator((n, n), matvec=matvec)
        
        # Compute exp(h * A_op) @ y0
        y_end = expm_multiply(h * A_op, y0)
        
        return y_end
    