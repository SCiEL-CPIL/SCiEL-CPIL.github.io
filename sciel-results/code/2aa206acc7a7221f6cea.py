import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115
    
    if q == 3 and n == 6 and radius == 1:
        N = 729
        
        def int_to_ternary(x, n):
            return [int(x // (3 ** i) % 3) for i in range(n)]
        
        def ternary_to_int(vec):
            res = 0
            for i, v in enumerate(vec):
                res += v * (3 ** i)
            return res

        balls = []
        for i in range(N):
            vec = int_to_ternary(i, n)
            ball = [i]
            for pos in range(n):
                orig = vec[pos]
                for new_val in range(3):
                    if new_val != orig:
                        new_vec = vec[:]
                        new_vec[pos] = new_val
                        ball.append(ternary_to_int(new_vec))
            balls.append(ball)
            
        balls_np = np.array(balls, dtype=np.int32)
        
        best_code = None
        best_size = N + 1
        
        while time.time() < deadline:
            covered = np.zeros(N, dtype=bool)
            gain = np.full(N, 13, dtype=np.int32)
            selected = np.zeros(N, dtype=bool)
            code = []
            
            for _ in range(N):
                if time.time() > deadline:
                    break
                
                valid_gain = np.where(selected, -1, gain)
                noise = np.random.uniform(0, 1, N)
                valid_gain_noisy = np.where(selected, -1, valid_gain + noise * 0.01)
                
                max_idx = np.argmax(valid_gain_noisy)
                
                if gain[max_idx] <= 0:
                    break
                    
                selected[max_idx] = True
                code.append(max_idx)
                
                ball = balls_np[max_idx]
                for u in ball:
                    if not covered[u]:
                        covered[u] = True
                        for v in balls_np[u]:
                            if not selected[v]:
                                gain[v] -= 1
                                
                if np.all(covered):
                    break
            
            if len(code) < best_size:
                best_size = len(code)
                best_code = code[:]
                
                # Reduction phase
                current_code = best_code[:]
                i = 0
                while i < len(current_code):
                    c = current_code[i]
                    temp_covered = np.zeros(N, dtype=bool)
                    for wc in current_code:
                        if wc != c:
                            temp_covered[balls_np[wc]] = True
                    
                    if np.all(temp_covered):
                        current_code.pop(i)
                        best_size -= 1
                    else:
                        i += 1
                
                best_code = current_code
                
                if best_size <= 71:
                    break
        
        if best_code is not None:
            return [int_to_ternary(idx, n) for idx in best_code]
        else:
            return []
    else:
        # Fallback for other parameters: Simple Greedy
        # This is not optimized for the specific task but ensures validity
        N = q ** n
        def int_to_q(x, n):
            return [int(x // (q ** i) % q) for i in range(n)]
        def q_to_int(vec):
            res = 0
            for i, v in enumerate(vec):
                res += v * (q ** i)
            return res

        balls = []
        for i in range(N):
            vec = int_to_q(i, n)
            ball = [i]
            for pos in range(n):
                orig = vec[pos]
                for new_val in range(q):
                    if new_val != orig:
                        new_vec = vec[:]
                        new_vec[pos] = new_val
                        ball.append(q_to_int(new_vec))
            balls.append(ball)
        balls_np = np.array(balls, dtype=np.int32)
        
        covered = np.zeros(N, dtype=bool)
        gain = np.array([len(b) for b in balls], dtype=np.int32)
        selected = np.zeros(N, dtype=bool)
        code = []
        
        while time.time() < deadline:
            valid_gain = np.where(selected, -1, gain)
            if np.max(valid_gain) <= 0: break
            max_idx = np.argmax(valid_gain)
            selected[max_idx] = True
            code.append(max_idx)
            for u in balls_np[max_idx]:
                if not covered[u]:
                    covered[u] = True
                    for v in balls_np[u]:
                        if not selected[v]:
                            gain[v] -= 1
            if np.all(covered): break
        
        return [int_to_q(idx, n) for idx in code]
