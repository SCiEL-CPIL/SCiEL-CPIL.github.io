import math
import time
import numpy as np

def construct():
    # Deadline guard
    deadline = time.time() + 590.0
    
    # Base Bessel J_{1/2} zeros: (k*pi)^2
    k = np.arange(1, 13)
    base_nodes = (k * math.pi) ** 2
    
    # We want to scale these nodes by 's' to minimize B.
    # B scales linearly with s, so we want the smallest valid s.
    # Validity is determined by the non-singularity of the linear system.
    # We approximate the condition number using floating point Laguerre evaluations.
    
    # Pre-compute Laguerre polynomial coefficients or use scipy?
    # scipy.special.eval_genlaguerre is available.
    from scipy.special import eval_genlaguerre
    
    def get_matrix(s):
        # Nodes: 0, s*u_1, ..., s*u_12
        # Conditions:
        # 1. g(0) = 0
        # 2. g'(0) = 1
        # 3. g(s*u_i) = 0
        # 4. g'(s*u_i) = 0
        # g(x) = sum_{k=0}^{25} c_k L_{2k}^{-1/2}(x)
        
        A = np.zeros((26, 26))
        
        # Row 0: g(0) = 0 -> sum c_k L_{2k}(0) = 0
        # Row 1: g'(0) = 1 -> sum c_k L'_{2k}(0) = 1 (RHS is 1, but we just check matrix condition)
        # L_{2k}^{-1/2}(0) = binom(2k - 0.5 + 1 - 1, 2k) = binom(2k + 0.5, 2k) ... 
        # Actually, L_n^{(alpha)}(0) = binom(n+alpha, n)
        # L_{2k}^{-1/2}(0) = binom(2k - 0.5, 2k)
        
        # We can just use eval_genlaguerre for values and derivatives.
        # Derivative: L_n^{(alpha)'}(x) = - L_{n-1}^{(alpha+1)}(x)
        
        # Row 0: values at x=0
        for k in range(26):
            A[0, k] = eval_genlaguerre(2*k, -0.5, 0.0)
            
        # Row 1: derivatives at x=0
        for k in range(26):
            # L'_{2k}^{-1/2}(0) = - L_{2k-1}^{0.5}(0)
            if 2*k == 0:
                A[1, k] = 0.0 # L_0 is constant 1, deriv 0
            else:
                A[1, k] = -eval_genlaguerre(2*k - 1, 0.5, 0.0)
                
        # Rows 2..25: for i in 1..12
        for i in range(1, 13):
            x_val = s * base_nodes[i-1]
            row_val = 1 + 2*(i-1)
            row_der = row_val + 1
            
            # g(x_val) = 0
            for k in range(26):
                A[row_val, k] = eval_genlaguerre(2*k, -0.5, x_val)
                
            # g'(x_val) = 0
            for k in range(26):
                if 2*k == 0:
                    A[row_der, k] = 0.0
                else:
                    A[row_der, k] = -eval_genlaguerre(2*k - 1, 0.5, x_val)
                    
        return A

    # Search for smallest s with good condition number
    best_s = 0.1 # Fallback
    best_cond = float('inf')
    
    # Try a log-spaced range of s
    # s too small -> singular. s too large -> B large.
    # We want smallest s.
    s_candidates = np.logspace(-4, -1, 30)
    
    for s in s_candidates:
        if time.time() > deadline:
            break
        A = get_matrix(s)
        # Check condition number
        # Use SVD for robustness
        try:
            cond = np.linalg.cond(A)
        except Exception:
            cond = float('inf')
            
        if cond < best_cond:
            best_cond = cond
            best_s = s
            
        # If condition number is "good enough", we can stop?
        # We want the SMALLEST s. The loop goes from small to large?
        # np.logspace(-4, -1) goes from 1e-4 to 1e-1.
        # So we start with small s.
        # If the first few are singular (inf), we move to larger s.
        # We want the first one that is NOT singular (or has low cond).
        if cond < 1e10:
            # Found a valid s. Since we iterate from small to large, this is the smallest valid one.
            # However, maybe a slightly larger s is more stable?
            # The score is proportional to s. So we want the absolute smallest valid s.
            # If cond is finite, it's valid.
            # Let's pick this s and break?
            # Risk: numerical noise might make it look valid when it's not for exact rational.
            # But 1e10 is a safe margin for double precision.
            best_s = s
            break
            
    # If no s was found with cond < 1e10, use the best (lowest cond) found
    # But we should ensure best_s is not 0.
    
    # Final nodes
    z = [float(s_val * base_node) for s_val, base_node in zip([best_s]*12, base_nodes)]
    
    # Safety checks
    if any(not (math.isfinite(zi) and zi > 0) for zi in z):
        # Fallback to unscaled Bessel nodes if scaling failed badly
        z = [float(base_node) for base_node in base_nodes]
        
    if len(set(z)) != 12:
        z = [float(base_node) for base_node in base_nodes]

    return sorted(z)
