import numpy as np

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    N = 64
    support = np.arange(N, dtype=np.float64)
    
    def compute_P(weights):
        # P = sum_{k,l} w_k w_l CDF_Y(2*s_k - s_l)
        P_Y = np.convolve(weights, weights, mode='full')
        CDF_Y = np.zeros(len(P_Y) + 1)
        CDF_Y[1:] = np.cumsum(P_Y)
        
        total = 0.0
        for k in range(N):
            s_k = support[k]
            for l in range(N):
                s_l = support[l]
                T = 2 * s_k - s_l
                if T <= 0:
                    val = 0.0
                elif T > len(P_Y):
                    val = 1.0 # CDF_Y[-1] is sum of P_Y which is 1 if weights sum to 1
                else:
                    # CDF_Y(T) = sum_{v < T} P_Y(v)
                    # P_Y index v corresponds to sum v.
                    # We want sum_{v=0}^{T-1} P_Y[v]
                    # CDF_Y array: CDF_Y[i] = sum_{v=0}^{i-1} P_Y[v]
                    # So we want CDF_Y[T]
                    val = CDF_Y[int(T)]
                total += weights[k] * weights[l] * val
        return total

    def compute_grad(weights):
        # grad_m = sum_l w_l CDF_Y(2*s_m - s_l) + sum_k w_k CDF_Y(2*s_k - s_m) + 2 * sum_k w_k W_prefix(2*s_k - s_m - 1)
        
        P_Y = np.convolve(weights, weights, mode='full')
        CDF_Y = np.zeros(len(P_Y) + 1)
        CDF_Y[1:] = np.cumsum(P_Y)
        
        # W_prefix[i] = sum_{j=0}^{i} weights[j]
        W_prefix = np.zeros(N)
        W_prefix[0] = weights[0]
        W_prefix[1:] = np.cumsum(weights[1:])
        
        grad = np.zeros(N)
        
        for m in range(N):
            s_m = support[m]
            
            # Term 1: sum_l w_l CDF_Y(2*s_m - s_l)
            t1 = 0.0
            for l in range(N):
                T = 2 * s_m - support[l]
                if T <= 0:
                    val = 0.0
                elif T > len(P_Y):
                    val = 1.0
                else:
                    val = CDF_Y[int(T)]
                t1 += weights[l] * val
            
            # Term 2: sum_k w_k CDF_Y(2*s_k - s_m)
            t2 = 0.0
            for k in range(N):
                T = 2 * support[k] - s_m
                if T <= 0:
                    val = 0.0
                elif T > len(P_Y):
                    val = 1.0
                else:
                    val = CDF_Y[int(T)]
                t2 += weights[k] * val
            
            # Term 3: 2 * sum_k w_k W_prefix(2*s_k - s_m - 1)
            t3 = 0.0
            for k in range(N):
                idx = int(2 * support[k] - s_m - 1)
                if idx < 0:
                    val = 0.0
                elif idx >= N:
                    val = 1.0 # sum of all weights
                else:
                    val = W_prefix[idx]
                t3 += weights[k] * val
            
            grad[m] = t1 + t2 + 2.0 * t3
        
        return grad

    def project_simplex(w):
        w = np.clip(w, 0, None)
        s = np.sum(w)
        if s > 0:
            w = w / s
        else:
            w = np.ones(N) / N
        return w

    best_weights = None
    best_P = -1.0
    
    # Multiple initializations to avoid local minima
    initializations = [
        np.ones(N) / N,
        np.array([0.5] * 2 + [0.0] * (N-2)),
        np.array([0.0] * (N-2) + [0.5] * 2),
        np.array([0.8] + [0.2] + [0.0]*(N-2)),
        np.array([0.0]*(N-2) + [0.8] + [0.2]),
        np.array([0.6, 0.2, 0.2] + [0.0]*(N-3)),
    ]
    
    for init_w in initializations:
        weights = project_simplex(init_w.copy())
        step = 0.1
        for it in range(200):
            grad = compute_grad(weights)
            P_old = compute_P(weights)
            
            alpha = step
            improved = False
            for _ in range(20):
                new_w = project_simplex(weights + alpha * grad)
                P_new = compute_P(new_w)
                if P_new > P_old + 1e-10:
                    weights = new_w
                    improved = True
                    break
                alpha *= 0.5
            
            if improved:
                P_cur = compute_P(weights)
                if P_cur > best_P:
                    best_P = P_cur
                    best_weights = weights.copy()
                step = min(step * 1.1, 0.5)
            else:
                step = max(step * 0.5, 0.001)
                if step < 0.001:
                    break
    
    return support, best_weights
