import time
import numpy as np

def construct():
    start_time = time.time()
    timeout = 110  # 10s margin
    
    # Generate primes up to 1000
    primes = [p for p in range(2, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
    n_primes = len(primes)
    
    # Residues, -1 means unassigned
    residues = [-1] * n_primes
    
    # Max L to try
    L_max = 2000
    
    # Coverage array, 1-indexed for integers 1..L_max
    cov = np.zeros(L_max + 1, dtype=np.int32)
    
    best_L = 0
    best_residues = residues.copy()
    
    for x in range(1, L_max + 1):
        if time.time() - start_time > timeout:
            break
            
        if cov[x] >= 2:
            best_L = x
            best_residues = residues.copy()
            continue
        
        # Need to fix x
        best_p_idx = -1
        best_loss = float('inf')
        best_priority = float('inf')
        best_p_val = float('inf')
        
        for i, p in enumerate(primes):
            target_r = x % p
            current_r = residues[i]
            
            if current_r == target_r:
                continue
                
            if current_r == -1:
                loss = 0
                priority = 0
            else:
                start = current_r
                if start == 0:
                    start = p
                if start >= x:
                    loss = 0
                else:
                    indices = np.arange(start, x, p)
                    loss = np.sum(cov[indices] == 2)
                priority = 1
            
            if (priority < best_priority) or \
               (priority == best_priority and loss < best_loss) or \
               (priority == best_priority and loss == best_loss and p < best_p_val):
                best_priority = priority
                best_loss = loss
                best_p_val = p
                best_p_idx = i
        
        if best_p_idx == -1:
            break
            
        old_r = residues[best_p_idx]
        new_r = x % primes[best_p_idx]
        p = primes[best_p_idx]
        
        if old_r != -1:
            start = old_r
            if start == 0:
                start = p
            if start <= L_max:
                indices = np.arange(start, L_max + 1, p)
                cov[indices] -= 1
        
        start = new_r
        if start == 0:
            start = p
        if start <= L_max:
            indices = np.arange(start, L_max + 1, p)
            cov[indices] += 1
        
        residues[best_p_idx] = new_r
        
        if best_loss > 0:
            break
        else:
            best_L = x
            best_residues = residues.copy()
            
    final_residues = []
    for r in best_residues:
        if r == -1:
            final_residues.append(0)
        else:
            final_residues.append(r)
            
    return final_residues
