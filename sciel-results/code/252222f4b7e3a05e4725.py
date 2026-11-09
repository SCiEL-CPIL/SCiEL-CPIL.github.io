import numpy as np
import random

def construct():
    N = 17
    D = 6
    MAX_CODE_SIZE = 1000 # Safe upper bound
    
    # Pre-allocate numpy array for code
    # We will store bits as uint8
    
    # Generate pool
    pool = [random.getrandbits(N) for _ in range(100000)]
    
    best_code = []
    best_len = 0
    
    for restart in range(20):
        random.shuffle(pool)
        
        # Numpy array for current code
        code_arr = np.zeros((MAX_CODE_SIZE, N), dtype=np.uint8)
        code_len = 0
        
        for w in pool:
            # Convert w to bits
            # Optimization: Pre-compute bits for pool? 
            # 100k * 17 bits = 1.7M bits. Memory is fine.
            # But converting int to array is slow.
            
            # Let's stick to Python bit_count for the inner check 
            # but use a list for the code.
            # If 300M ops is too slow, we need NumPy.
            
            pass # Placeholder
            
    # I will use the Python bit_count version but with a smaller pool or fewer restarts if it's too slow.
    # Actually, 100k candidates * 20 restarts = 2M candidates.
    # Average code size 250.
    # 2M * 250 = 500M distance checks.
    # In Python, 500M loops is likely > 60s.
    
    # So I MUST use NumPy or reduce the pool.
    
    # Let's use a pool of 20,000 and 50 restarts.
    # 20k * 50 = 1M candidates.
    # 1M * 250 = 250M checks. Still risky.
    
    # Let's use NumPy properly.
    
    best_code = []
    best_len = 0
    
    pool = [random.getrandbits(N) for _ in range(50000)]
    
    for restart in range(30):
        random.shuffle(pool)
        
        code_list = []
        # Numpy array for fast distance calc
        # We'll maintain a list of numpy arrays? No.
        # We'll convert the current code to numpy array only when we add a new one?
        # No, we need to check against ALL existing.
        
        # Strategy:
        # 1. Keep `code_list` (ints).
        # 2. Keep `code_np` (numpy array).
        # 3. When adding, `code_np = np.vstack([code_np, new_bit_array])`.
        # `vstack` is O(K). 300 * 300 = 90k ops per restart? No, 300 additions * O(K) copy.
        # Total copy cost: Sum(K) ~ 300^2 / 2 = 45k ops. Negligible.
        
        code_np = np.empty((0, N), dtype=np.uint8)
        
        for w in pool:
            # Convert w to numpy
            w_np = np.array([(w >> i) & 1 for i in range(N)], dtype=np.uint8)
            
            if code_np.shape[0] == 0:
                code_list.append(w)
                code_np = w_np.reshape(1, N)
                continue
            
            # Distance check
            diffs = code_np ^ w_np
            min_dist = np.min(np.sum(diffs, axis=1))
            
            if min_dist >= D:
                code_list.append(w)
                code_np = np.vstack((code_np, w_np))
                
        if len(code_list) > best_len:
            best_len = len(code_list)
            best_code = code_list[:]
            
    return best_code
