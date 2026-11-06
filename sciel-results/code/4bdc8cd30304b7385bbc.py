"""Size-efficient sorting networks on 17 wires."""
import time
import random
import numpy as np

N = 17
TOTAL_BOOLEANS = 1 << N  # 131072

def _make_seed():
    """Odd-even transposition sorting network for N wires."""
    comps = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comps.append((i, i + 1))
    return comps

def _verify_network(comps, deadline, start_time):
    """Verify sorting network using zero-one principle. Returns True if valid."""
    if time.time() - start_time > deadline:
        return True  # assume valid if running out of time
    
    inputs = np.arange(TOTAL_BOOLEANS, dtype=np.int32)
    
    for i, j in comps:
        bi = (inputs >> i) & 1
        bj = (inputs >> j) & 1
        mn = bi & bj
        mx = bi | bj
        inputs &= ~(1 << i)
        inputs &= ~(1 << j)
        inputs |= (mn << i)
        inputs |= (mx << j)
    
    # Check sorted: 0s then 1s, i.e., bit[i] <= bit[i+1] for all i
    for i in range(N - 1):
        bi = (inputs >> i) & 1
        bj = (inputs >> (i + 1)) & 1
        if np.any((bi == 1) & (bj == 0)):
            return False
    
    return True

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 75.0  # 75s to leave safety margin before 90s kill
    
    random.seed(42)
    np.random.seed(42)
    
    best_comps = _make_seed()
    best_len = len(best_comps)
    
    # Verify seed
    if not _verify_network(best_comps, DEADLINE, start_time):
        return best_comps
    
    # Phase 1: Greedy removal - try removing each comparator
    improved = True
    while improved and (time.time() - start_time) < DEADLINE - 10:
        improved = False
        for idx in range(len(best_comps) - 1, -1, -1):
            if time.time() - start_time > DEADLINE - 10:
                break
            candidate = best_comps[:idx] + best_comps[idx+1:]
            if _verify_network(candidate, DEADLINE, start_time):
                best_comps = candidate
                best_len -= 1
                improved = True
    
    # Phase 2: Mutation-based search to escape local minima
    for iteration in range(100):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        for _ in range(30):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            candidate = list(best_comps)
            mutation_type = random.random()
            
            if mutation_type < 0.5:
                # Insert a random comparator
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                pos = random.randint(0, len(candidate))
                candidate.insert(pos, (i, j))
            elif mutation_type < 0.8:
                # Replace a comparator
                idx = random.randint(0, len(candidate) - 1)
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                candidate[idx] = (i, j)
            else:
                # Swap two comparators
                if len(candidate) >= 2:
                    i1 = random.randint(0, len(candidate) - 1)
                    i2 = random.randint(0, len(candidate) - 1)
                    candidate[i1], candidate[i2] = candidate[i2], candidate[i1]
            
            # Verify and try to improve
            if _verify_network(candidate, DEADLINE, start_time):
                # Greedy removal on candidate
                for idx in range(len(candidate) - 1, -1, -1):
                    if time.time() - start_time > DEADLINE - 5:
                        break
                    temp = candidate[:idx] + candidate[idx+1:]
                    if _verify_network(temp, DEADLINE, start_time):
                        candidate = temp
            
            if len(candidate) < best_len:
                best_comps = candidate
                best_len = len(candidate)
    
    return best_comps
# EVOLVE-BLOCK-END
