import numpy as np
import time

def construct():
        # 1. Get primes
        primes = [p for p in range(2, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
        n_primes = len(primes)
        
        # 2. Setup
        M_MAX = 2000
        x = np.arange(1, M_MAX + 1)
        
        # 3. Initialization
        res = np.random.randint(0, 1000, size=n_primes)
        for i, p in enumerate(primes):
            res[i] %= p
            
        # 4. Evaluation Function
        def evaluate(res):
            counts = np.zeros(M_MAX, dtype=np.int8)
            for i, p in enumerate(primes):
                r = res[i]
                # Vectorized modulo check
                # x % p == r
                counts += (x % p == r)
            
            # Find L: largest L such that counts[0:L] >= 2
            # counts[0] corresponds to x=1
            # We need the first index where counts < 2
            # If all >= 2, L = M_MAX
            # Use np.where
            bad = np.where(counts < 2)[0]
            if len(bad) == 0:
                L = M_MAX
            else:
                L = bad[0]
            
            # Tie breaker: sum of counts in prefix
            # To make it fast, just use L primarily
            # But adding a small fraction of robustness helps escape plateaus
            robustness = np.sum(counts[:L]) if L > 0 else 0
            score = L * 1000 + robustness
            return score, L

        # 5. Simulated Annealing
        start_time = time.time()
        time_limit = 110 # Safety margin
        
        current_score, current_L = evaluate(res)
        best_res = res.copy()
        best_score = current_score
        best_L = current_L
        
        T = 1000.0
        cooling_rate = 0.999
        
        while time.time() - start_time < time_limit:
            # Move
            i = np.random.randint(n_primes)
            p = primes[i]
            old_r = res[i]
            new_r = np.random.randint(p)
            
            if new_r == old_r:
                continue
                
            res[i] = new_r
            new_score, new_L = evaluate(res)
            
            delta = new_score - current_score
            if delta > 0 or np.random.random() < np.exp(delta / T):
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_L = new_L
                    best_res = res.copy()
            else:
                res[i] = old_r
                
            T *= cooling_rate
            if T < 1e-3:
                break
                
        return best_res.tolist()
