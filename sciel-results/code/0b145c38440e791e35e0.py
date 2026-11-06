import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    N = 512
    K = 32
    
    def score_bitset(b):
        """Count distinct d > 0 such that a, a+d, a+2d all in set (bit x-1 for x)."""
        count = 0
        # d ranges 1..255 (since max element 512, 2d <= 511)
        for d in range(1, 256):
            if (b & (b >> d) & (b >> (2 * d))):
                count += 1
        return count

    def set_to_bitset(S):
        b = 0
        for x in S:
            b |= (1 << (x - 1))
        return b

    def bitset_to_set(b):
        s = []
        for i in range(1, N + 1):
            if (b >> (i - 1)) & 1:
                s.append(i)
        return s

    def greedy_initialize(core_size=16):
        """
        Initialize a set using a greedy heuristic.
        Start with a dense core [1..core_size], then greedily add elements
        that maximize the score increase.
        """
        S = set(range(1, core_size + 1))
        b = set_to_bitset(list(S))
        current_score = score_bitset(b)
        
        # Candidates: numbers from core_size+1 to N
        candidates = list(range(core_size + 1, N + 1))
        random.shuffle(candidates)
        
        # Greedy addition
        while len(S) < K:
            best_candidate = None
            best_score = current_score
            best_delta = 0
            
            # Sample a subset of candidates to keep it fast, or iterate all if K is small
            # Since K=32, we have 16 to add. Iterating 500 candidates 16 times is fine.
            for c in candidates:
                if c in S:
                    continue
                # Test adding c
                new_b = b | (1 << (c - 1))
                new_score = score_bitset(new_b)
                delta = new_score - current_score
                if delta > best_delta:
                    best_delta = delta
                    best_score = new_score
                    best_candidate = c
            
            if best_candidate is not None:
                S.add(best_candidate)
                b |= (1 << (best_candidate - 1))
                current_score = best_score
            else:
                # If no improvement, just pick a random one to fill
                available = [c for c in candidates if c not in S]
                if available:
                    c = random.choice(available)
                    S.add(c)
                    b |= (1 << (c - 1))
                    current_score = score_bitset(b)
                else:
                    break
        
        return list(S)

    def triple_swap_sa(init_list, time_budget):
        """
        Simulated Annealing with Triple-Swap moves.
        State: list of K elements.
        Move: Pick 3 elements to remove, pick 3 new elements to add.
        """
        current_set = set(init_list)
        current_list = list(init_list)
        current_bit = set_to_bitset(current_list)
        current_score = score_bitset(current_bit)
        
        best_set = set(current_list)
        best_bit = current_bit
        best_score = current_score
        
        T = 10.0
        T_min = 0.1
        cooling = 0.9999
        all_nums = list(range(1, N + 1))
        
        while time.time() < time_budget and T > T_min:
            # Pick 3 indices to remove
            idxs = random.sample(range(K), 3)
            old_elems = [current_list[i] for i in idxs]
            
            # Pick 3 new elements not in current_set
            new_elems = []
            while len(new_elems) < 3:
                c = random.choice(all_nums)
                if c not in current_set and c not in new_elems:
                    new_elems.append(c)
            
            # Perform swap
            # Remove old
            for e in old_elems:
                current_set.remove(e)
                current_bit &= ~(1 << (e - 1))
            # Add new
            for e in new_elems:
                current_set.add(e)
                current_bit |= (1 << (e - 1))
            
            # Update list (order doesn't matter for score, but for state tracking)
            for i, e in zip(idxs, new_elems):
                current_list[i] = e
                
            new_score = score_bitset(current_bit)
            delta = new_score - current_score
            
            if delta > 0 or random.random() < math.exp(delta / T):
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_bit = current_bit
                    best_set = set(current_set)
            else:
                # Revert
                for e in new_elems:
                    current_set.remove(e)
                    current_bit &= ~(1 << (e - 1))
                for i, e in zip(idxs, old_elems):
                    current_set.add(e)
                    current_bit |= (1 << (e - 1))
                    current_list[i] = e
            
            T *= cooling
            
        return best_set, best_bit, best_score

    def single_swap_sa(init_list, time_budget):
        """
        Simulated Annealing with Single-Swap moves.
        Used for fine-tuning.
        """
        current_set = set(init_list)
        current_list = list(init_list)
        current_bit = set_to_bitset(current_list)
        current_score = score_bitset(current_bit)
        
        best_set = set(current_list)
        best_bit = current_bit
        best_score = current_score
        
        T = 5.0
        T_min = 0.1
        cooling = 0.9999
        all_nums = list(range(1, N + 1))
        
        while time.time() < time_budget and T > T_min:
            idx = random.randint(0, K - 1)
            old_elem = current_list[idx]
            
            while True:
                new_elem = random.choice(all_nums)
                if new_elem not in current_set:
                    break
            
            current_set.remove(old_elem)
            current_bit &= ~(1 << (old_elem - 1))
            current_set.add(new_elem)
            current_bit |= (1 << (new_elem - 1))
            current_list[idx] = new_elem
            
            new_score = score_bitset(current_bit)
            delta = new_score - current_score
            
            if delta > 0 or random.random() < math.exp(delta / T):
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_bit = current_bit
                    best_set = set(current_set)
            else:
                current_set.remove(new_elem)
                current_bit &= ~(1 << (new_elem - 1))
                current_set.add(old_elem)
                current_bit |= (1 << (old_elem - 1))
                current_list[idx] = old_elem
                
            T *= cooling
            
        return best_set, best_bit, best_score

    best_overall_score = 0
    best_overall_bit = 0
    best_overall_set = []

    # Phase 1: Multi-restart with Greedy Init + Triple Swap SA
    num_restarts = 4
    time_per_restart = (deadline - time.time() - 10.0) / (num_restarts + 1)
    
    for i in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Generate a diverse greedy init
        # Vary the core size or the random seed for candidates
        random.seed(time.time_ns() + i)
        init_list = greedy_initialize(core_size=12 + i * 2)
        
        run_deadline = time.time() + time_per_restart
        best_set, best_bit, best_score = triple_swap_sa(init_list, run_deadline)
        
        if best_score > best_overall_score:
            best_overall_score = best_score
            best_overall_bit = best_bit
            best_overall_set = list(best_set)

    # Phase 2: Fine-tuning with Single Swap SA on the best result
    if best_overall_set and time.time() < deadline - 5.0:
        final_deadline = time.time() + (deadline - time.time() - 5.0)
        final_set, final_bit, final_score = single_swap_sa(best_overall_set, final_deadline)
        if final_score > best_overall_score:
            best_overall_score = final_score
            best_overall_bit = final_bit
            best_overall_set = list(final_set)

    # Ensure valid output
    result = sorted(best_overall_set) if best_overall_set else list(range(1, 33))
    
    # Validation
    if len(result) != K:
        result = list(range(1, 33))
    if len(set(result)) != K:
        result = list(range(1, 33))
    if not all(1 <= x <= N for x in result):
        result = list(range(1, 33))
        
    return result

# EVOLVE-BLOCK-END
