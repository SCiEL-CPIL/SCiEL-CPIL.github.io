import numpy as np
import time

def construct():
    # Generate GRS sequence
    p, q = [1], [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    best_c = np.array(p[:24])
    
    # Precompute sampling matrix
    M = 50000
    theta = np.linspace(0, 2 * np.pi, M, endpoint=False)
    z = np.exp(1j * theta)
    k = np.arange(24)
    Z = z[:, None] ** k[None, :]
    
    def sup_norm(c):
        P = Z @ c
        return np.max(np.abs(P))
    
    best_score = sup_norm(best_c)
    
    start_time = time.time()
    deadline = 110.0
    
    # Local search with single bit flips
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        for k in range(24):
            c_new = best_c.copy()
            c_new[k] *= -1
            score_new = sup_norm(c_new)
            if score_new < best_score:
                best_c = c_new
                best_score = score_new
                improved = True
                
    # Try a few random restarts if time permits
    while (time.time() - start_time) < deadline:
        c_rand = np.random.choice([-1, 1], size=24)
        score_rand = sup_norm(c_rand)
        if score_rand < best_score:
            best_c = c_rand
            best_score = score_rand
            improved = True
            while improved and (time.time() - start_time) < deadline:
                improved = False
                for k in range(24):
                    c_new = best_c.copy()
                    c_new[k] *= -1
                    score_new = sup_norm(c_new)
                    if score_new < best_score:
                        best_c = c_new
                        best_score = score_new
                        improved = True
        else:
            # If no improvement from random, we might be near optimal
            # but let's try a few more
            pass

    return list(best_c)
