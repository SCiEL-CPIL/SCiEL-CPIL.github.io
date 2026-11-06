import numpy as np
import time
import random

def construct():
        start_time = time.time()
        limit = 55 # 5s buffer
        
        def objective(A):
            A = np.array(A)
            sums = A[:, None] + A[None, :]
            prods = A[:, None] * A[None, :]
            return max(len(np.unique(sums)), len(np.unique(prods)))
        
        def is_valid(A):
            if len(set(A)) != 20: return False
            if any(x < 1 or x > 10000 for x in A): return False
            return True
        
        best_val = float('inf')
        best_A = None
        
        # Initialize with various structures
        candidates = []
        
        # 1. APs
        for _ in range(20):
            a = random.randint(1, 100)
            d = random.randint(1, 10)
            A = [a + i*d for i in range(20)]
            if is_valid(A): candidates.append(A)
            
        # 2. GPs
        for _ in range(20):
            a = random.randint(1, 10)
            r = random.randint(2, 5)
            A = []
            cur = a
            while len(A) < 20 and cur <= 10000:
                A.append(cur)
                cur *= r
            if len(A) == 20 and is_valid(A): candidates.append(A)
            
        # 3. Random
        for _ in range(20):
            A = random.sample(range(1, 10001), 20)
            candidates.append(A)
            
        # 4. Hybrid (Union of two APs? or AP + noise)
        
        for A in candidates:
            if time.time() - start_time > limit: break
            A = list(A)
            cur_val = objective(A)
            if cur_val < best_val:
                best_val = cur_val
                best_A = list(A)
                
            # Local Search
            temp = 10.0
            cooling = 0.99
            while temp > 0.1 and (time.time() - start_time < limit):
                idx = random.randint(0, 19)
                old = A[idx]
                
                # Generate new value
                # Bias towards values that create collisions?
                # Just random for now, maybe local neighborhood
                if random.random() < 0.5:
                    new_val = old + random.randint(-50, 50)
                else:
                    new_val = random.randint(1, 10000)
                    
                if new_val < 1 or new_val > 10000 or new_val in A:
                    continue
                    
                A[idx] = new_val
                new_val_obj = objective(A)
                delta = new_val_obj - cur_val
                
                if delta < 0 or random.random() < np.exp(-delta / temp):
                    A[idx] = new_val
                    cur_val = new_val_obj
                    if cur_val < best_val:
                        best_val = cur_val
                        best_A = list(A)
                else:
                    A[idx] = old
                    
                temp *= cooling
                
        return sorted(best_A)
