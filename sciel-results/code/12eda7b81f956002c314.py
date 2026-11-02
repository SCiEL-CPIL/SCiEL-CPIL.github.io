"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct(q, n, radius):
    start_time = time.time()
    timeout = 145.0  # Leave 5s margin for safety
    
    total_words = q ** n
    
    # Precompute powers of q for digit extraction
    powers = [q ** i for i in range(n)]
    
    # Convert between integer representation and digit lists
    def int_to_digits(x):
        digits = [0] * n
        for i in range(n):
            digits[i] = x % q
            x //= q
        return digits

    def digits_to_int(digits):
        val = 0
        for i in range(n):
            val = val * q + digits[i]
        return val

    # Precompute neighbors for all words (distance <= radius)
    # For radius=1, each word has 1 + n*(q-1) neighbors
    neighbors = [[] for _ in range(total_words)]
    for u in range(total_words):
        neighbors[u].append(u)  # self
        du = int_to_digits(u)
        for pos in range(n):
            orig_val = du[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                v = u - orig_val * powers[pos] + new_val * powers[pos]
                neighbors[u].append(v)
                
    # 1. Greedy Initialization with Randomization
    best_code = []
    best_size = total_words + 1
    
    max_restarts = 50
    for restart in range(max_restarts):
        if time.time() - start_time > timeout - 15: # Reserve time for local search
            break
            
        is_uncovered = [True] * total_words
        uncovered_count = total_words
        counts = [0] * total_words
        C = []
        
        while uncovered_count > 0:
            best_u = -1
            best_gain = -1
            max_possible_gain = len(neighbors[0])
            
            candidates = list(range(total_words))
            random.shuffle(candidates)
            
            for u in candidates:
                gain = 0
                for v in neighbors[u]:
                    if is_uncovered[v]:
                        gain += 1
                        if gain + best_gain > max_possible_gain: # Early break? No, gain is per u
                            pass
                            
                # Add small random noise
                noise = random.random() * 0.01
                if gain + noise > best_gain:
                    best_gain = gain + noise
                    best_u = u
                    
            if best_u == -1:
                break
                
            C.append(best_u)
            for v in neighbors[best_u]:
                counts[v] += 1
                if is_uncovered[v]:
                    is_uncovered[v] = False
                    uncovered_count -= 1
            
        # Local Search: Remove redundant words
        removed_any = True
        while removed_any:
            removed_any = False
            random.shuffle(C)
            i = 0
            while i < len(C):
                w = C[i]
                safe = True
                for v in neighbors[w]:
                    if counts[v] == 1:
                        safe = False
                        break
                if safe:
                    for v in neighbors[w]:
                        counts[v] -= 1
                    C.pop(i)
                    removed_any = True
                else:
                    i += 1
        
        if len(C) < best_size:
            best_size = len(C)
            best_code = C.copy()
            
    # Rebuild counts for best_code
    counts = [0] * total_words
    for w in best_code:
        for v in neighbors[w]:
            counts[v] += 1
            
    uncovered_count = sum(1 for c in counts if c == 0)
    
    # 2. Simulated Annealing / Local Search
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.9995
    
    current_code = best_code
    current_size = len(current_code)
    current_uncovered = uncovered_count
    current_counts = counts
    
    best_sa_code = current_code.copy()
    best_sa_size = current_size
    best_sa_counts = current_counts.copy()
    
    # If we started with a valid code, current_uncovered should be 0
    if current_uncovered > 0:
         # Fallback: just return best_code
         result = [int_to_digits(c) for c in best_code]
         return result

    while time.time() - start_time < timeout - 5:
        if T < T_min:
            T = T_min
            
        if current_size > 0:
            i = random.randint(0, current_size - 1)
            w_old = current_code[i]
            w_new = random.randint(0, total_words - 1)
            
            if w_old != w_new:
                # Remove w_old
                delta_uncovered = 0
                for v in neighbors[w_old]:
                    current_counts[v] -= 1
                    if current_counts[v] == 0:
                        delta_uncovered += 1
                
                # Add w_new
                for v in neighbors[w_new]:
                    if current_counts[v] == 0:
                        delta_uncovered -= 1
                    current_counts[v] += 1
                
                new_uncovered = current_uncovered + delta_uncovered
                
                accept = False
                if new_uncovered < current_uncovered:
                    accept = True
                elif new_uncovered == current_uncovered:
                    accept = random.random() < 0.05
                else:
                    delta_cost = new_uncovered - current_uncovered
                    if delta_cost > 0:
                        prob = math.exp(-delta_cost / T)
                        accept = random.random() < prob
                
                if accept:
                    current_code[i] = w_new
                    current_uncovered = new_uncovered
                    
                    # Reduce size
                    removed_any = True
                    while removed_any:
                        removed_any = False
                        random.shuffle(current_code)
                        j = 0
                        while j < len(current_code):
                            w = current_code[j]
                            safe = True
                            for v in neighbors[w]:
                                if current_counts[v] == 1:
                                    safe = False
                                    break
                            if safe:
                                for v in neighbors[w]:
                                    current_counts[v] -= 1
                                current_code.pop(j)
                                removed_any = True
                            else:
                                j += 1
                            
                    if current_uncovered == 0 and len(current_code) < best_sa_size:
                        best_sa_size = len(current_code)
                        best_sa_code = current_code.copy()
                        best_sa_counts = current_counts.copy()
                        
                else:
                    # Rollback
                    for v in neighbors[w_old]:
                        current_counts[v] += 1
                    for v in neighbors[w_new]:
                        current_counts[v] -= 1
                        
        T *= cooling_rate
        
    result = [int_to_digits(c) for c in best_sa_code]
    return result
# EVOLVE-BLOCK-END
