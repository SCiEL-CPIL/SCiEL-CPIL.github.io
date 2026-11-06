"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

N = 512

# Precompute all 4-term APs and for each element, the triples that complete a 4-AP with it
_ap_others = {}

def _precompute():
    global _ap_others
    if _ap_others:
        return
    _ap_others = {i: [] for i in range(1, N + 1)}
    for a in range(1, N + 1):
        max_d = (N - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            _ap_others[a].append((b, c, e))
            _ap_others[b].append((a, c, e))
            _ap_others[c].append((a, b, e))
            _ap_others[e].append((a, b, c))

def _can_add(x, bits):
    """Check if adding x to set represented by bits creates a 4-AP."""
    for t0, t1, t2 in _ap_others[x]:
        if bits[t0] and bits[t1] and bits[t2]:
            return False
    return True

def _set_to_bits(S):
    bits = np.zeros(N + 1, dtype=bool)
    for x in S:
        bits[x] = True
    return bits

def _ternary_01_set(n):
    """Generate all numbers in [1, n] whose base-3 representation contains only 0s and 1s."""
    result = set()
    # 3^6 = 729 > 512, so we need up to 6 digits
    for mask in range(1, 1 << 6):
        val = 0
        for bit in range(6):
            if mask & (1 << bit):
                val += 3 ** bit
        if 1 <= val <= n:
            result.add(val)
    return result

def _greedy_construct(S_init, order, deadline):
    """Greedy construction starting from S_init, adding elements in 'order'."""
    S = set(S_init)
    S_bits = _set_to_bits(S)
    for x in order:
        if time.time() > deadline:
            break
        if x in S:
            continue
        if _can_add(x, S_bits):
            S.add(x)
            S_bits[x] = True
    return S

def _local_search(S, deadline, max_rounds=5000):
    """
    Local search: try to remove one element and add two others to increase size.
    Also includes 1-1 swaps for diversification.
    """
    S = set(S)
    S_bits = _set_to_bits(S)
    current_size = len(S)
    no_improvement_count = 0
    
    for _ in range(max_rounds):
        if time.time() > deadline:
            break
            
        # Try 1-2 move: remove 1, add 2
        if not S:
            break
        to_remove = random.choice(list(S))
        S_minus = S - {to_remove}
        S_minus_bits = S_bits.copy()
        S_minus_bits[to_remove] = False
        
        # Try to find 2 elements to add
        candidates = [x for x in range(1, N + 1) if x not in S_minus]
        random.shuffle(candidates)
        
        added = []
        temp_bits = S_minus_bits.copy()
        for c in candidates:
            if len(added) == 2:
                break
            if _can_add(c, temp_bits):
                added.append(c)
                temp_bits[c] = True
                
        if len(added) == 2:
            S = S_minus | set(added)
            S_bits = temp_bits
            current_size = len(S)
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count > 30:
                # Try a 1-1 swap to diversify
                to_remove = random.choice(list(S))
                S_minus = S - {to_remove}
                S_minus_bits = S_bits.copy()
                S_minus_bits[to_remove] = False
                candidates = [x for x in range(1, N + 1) if x not in S_minus]
                random.shuffle(candidates)
                found_swap = False
                for c in candidates:
                    if _can_add(c, S_minus_bits):
                        S = S_minus | {c}
                        S_bits = S_minus_bits
                        S_bits[c] = True
                        found_swap = True
                        break
                if not found_swap:
                    break
                no_improvement_count = 0
                
    return S

def _kick_and_repair(S, deadline):
    """
    Random kick: remove a few random elements, then repair greedily.
    This helps escape local minima.
    """
    S = set(S)
    if not S:
        return S
    
    # Remove 1-3 random elements
    kick_size = random.randint(1, 3)
    to_remove = random.sample(list(S), min(kick_size, len(S)))
    S_kicked = S - set(to_remove)
    
    # Repair: greedily add elements back
    order = list(range(1, N + 1))
    random.shuffle(order)
    S_repaired = _greedy_construct(S_kicked, order, deadline)
    
    # Local search on the repaired set
    S_final = _local_search(S_repaired, deadline, max_rounds=500)
    
    return S_final

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # 8s safety margin before 90s kill
    _precompute()
    random.seed(42)
    
    # Initial sets
    best_set = _ternary_01_set(N)
    best_size = len(best_set)
    
    # 1. Greedy from empty with random order
    order = list(range(1, N + 1))
    random.shuffle(order)
    greedy_set = _greedy_construct(set(), order, deadline)
    if len(greedy_set) > best_size:
        best_set = greedy_set
        best_size = len(best_set)
        
    # 2. Greedy from ternary
    order2 = list(range(1, N + 1))
    random.shuffle(order2)
    ternary_greedy = _greedy_construct(_ternary_01_set(N), order2, deadline)
    if len(ternary_greedy) > best_size:
        best_set = ternary_greedy
        best_size = len(best_set)
        
    # 3. Local search on best
    if time.time() < deadline:
        best_set = _local_search(best_set, deadline, max_rounds=3000)
        best_size = len(best_set)
        
    # 4. Multiple restarts with kicks
    for attempt in range(50):
        if time.time() > deadline - 2.0:
            break
            
        # Random start
        if attempt % 3 == 0:
            S_init = set()
        elif attempt % 3 == 1:
            S_init = _ternary_01_set(N)
        else:
            S_init = set(random.sample(range(1, N + 1), 30))
            
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        current = _greedy_construct(S_init, order, deadline)
        
        # Local search
        if len(current) >= best_size:
            current = _local_search(current, deadline, max_rounds=1000)
        else:
            # Even if not better, do a kick-and-repair to explore
            current = _kick_and_repair(current, deadline)
            
        if len(current) > best_size:
            best_set = current
            best_size = len(current)
            
    # 5. Final kick-and-repair on the best set
    if time.time() < deadline - 5.0:
        kicked = _kick_and_repair(best_set, deadline)
        if len(kicked) > best_size:
            best_set = kicked
            best_size = len(best_set)
    
    # 6. Final local search polish
    if time.time() < deadline - 1.0:
        polished = _local_search(best_set, deadline, max_rounds=2000)
        if len(polished) > best_size:
            best_set = polished
            best_size = len(best_set)
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
