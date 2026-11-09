import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    n = 12
    m = 40
    N = 1 << n # 4096
    
    # Precompute comparability matrix
    # I, J shape (N, 1) and (1, N)
    I = np.arange(N)[:, None]
    J = np.arange(N)[None, :]
    # (I & J) == I is I subset J
    # (J & I) == J is J subset I
    Comp = ((I & J) == I) | ((J & I) == J)
    np.fill_diagonal(Comp, False)
    
    best_score = -1
    best_set = np.arange(m) # default init
    
    def local_search(init_set):
        S = init_set.copy()
        S_indices = S
        current_score = np.sum(Comp[S_indices][:, S_indices])
        
        while time.time() < deadline:
            # Calculate Gains for all nodes
            # Gains[x] = sum(Comp[x, S])
            Gains = Comp[:, S_indices].sum(axis=1)
            
            best_delta = 0
            best_a_idx = -1
            best_b = -1
            
            # Try swapping each a in S with best b not in S
            for idx, a in enumerate(S):
                # Va[b] = Gains[b] - Comp[b, a]
                Va = Gains - Comp[:, a]
                # Mask out current set
                Va[S_indices] = -np.inf
                
                b = np.argmax(Va)
                delta = Va[b] - Gains[a]
                
                if delta > best_delta:
                    best_delta = delta
                    best_a_idx = idx
                    best_b = b
            
            if best_delta > 0:
                # Perform swap
                S[best_a_idx] = best_b
                S_indices = S
                current_score += best_delta
            else:
                break
        
        return S, current_score

    # Initialization strategies
    inits = [np.arange(m)]
    
    # Random inits from small universes
    for _ in range(5):
        inits.append(np.random.choice(64, m, replace=False))
    for _ in range(5):
        inits.append(np.random.choice(128, m, replace=False))
        
    for init in inits:
        if time.time() > deadline:
            break
        S, score = local_search(init)
        if score > best_score:
            best_score = score
            best_set = S

    # Convert to lists
    result = []
    for mask in best_set:
        result.append([b for b in range(n) if (mask >> b) & 1])
        
    return result
