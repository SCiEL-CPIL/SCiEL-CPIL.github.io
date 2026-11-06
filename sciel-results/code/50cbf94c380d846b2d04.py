"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # Bitset representation: x in 1..512 maps to bit (x-1)
    # S is an integer where bit i is 1 if (i+1) is in the set

    def bitset_from_set(S):
        b = 0
        for x in S:
            b |= (1 << (x - 1))
        return b

    def set_from_bitset(b):
        s = []
        i = 0
        while b:
            if b & 1:
                s.append(i + 1)
            b >>= 1
            i += 1
        return s

    def score_bitset(b):
        count = 0
        for d in range(1, 257):
            if (b & (b >> d) & (b >> (2 * d))):
                count += 1
        return count

    def popcount(b):
        return bin(b).count('1')

    def random_set():
        s = set(random.sample(range(1, 513), 32))
        return s

    def greedy_construct():
        """Greedy: add elements that maximize new AP differences."""
        S = set()
        # Start with a random seed of a few elements
        for _ in range(4):
            S.add(random.randint(1, 512))
        
        while len(S) < 32:
            best_elem = None
            best_new_score = -1
            current_score = score_bitset(bitset_from_set(S))
            
            candidates = list(range(1, 513))
            random.shuffle(candidates)
            
            for elem in candidates:
                if elem in S:
                    continue
                new_S = S | {elem}
                new_score = score_bitset(bitset_from_set(new_S))
                if new_score > best_new_score:
                    best_new_score = new_score
                    best_elem = elem
            
            if best_elem is not None:
                S.add(best_elem)
        
        return S

    def local_search(S, time_limit):
        """Hill climbing with swap moves."""
        s = set(S)
        b = bitset_from_set(s)
        current_score = score_bitset(b)
        best_score = current_score
        best_set = set(s)
        
        end_time = time.time() + time_limit
        no_improve_count = 0
        
        while time.time() < end_time:
            # Try a random swap
            elems = list(s)
            removed = random.choice(elems)
            added = random.randint(1, 512)
            
            if added == removed or added in s:
                continue
            
            new_s = set(s)
            new_s.remove(removed)
            new_s.add(added)
            
            new_b = bitset_from_set(new_s)
            new_score = score_bitset(new_b)
            
            if new_score > current_score:
                s = new_s
                b = new_b
                current_score = new_score
                no_improve_count = 0
                
                if current_score > best_score:
                    best_score = current_score
                    best_set = set(s)
            else:
                no_improve_count += 1
                # Allow some non-improving moves to escape local optima
                if no_improve_count > 200 and random.random() < 0.1:
                    s = new_s
                    b = new_b
                    current_score = new_score
                    no_improve_count = 0
        
        return best_set, best_score

    # Strategy: multiple approaches
    best_set = None
    best_score = -1

    # Approach 1: Greedy construction + local search
    if time.time() < deadline - 30:
        gS = greedy_construct()
        gS2, gscore = local_search(gS, min(20, deadline - time.time() - 5))
        if gscore > best_score:
            best_score = gscore
            best_set = gS2

    # Approach 2: Random restarts with local search
    num_restarts = 30
    for restart in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Generate initial set
        if restart == 0:
            # Start with a structured set (arithmetic progression-like)
            s = set()
            for base in range(1, 513, 16):
                for step in range(1, 16):
                    candidate = set()
                    for i in range(32):
                        val = base + i * step
                        if val <= 512:
                            candidate.add(val)
                    if len(candidate) == 32:
                        s = candidate
                        break
                if len(s) == 32:
                    break
            if len(s) != 32:
                s = random_set()
        elif restart == 1:
            # Sparse structured set
            s = set()
            for i in range(32):
                s.add(1 + i * 16)
        else:
            s = random_set()
        
        time_per_search = min(5.0, (deadline - 10.0 - time.time()) / max(1, num_restarts - restart))
        s2, score = local_search(s, time_per_search)
        
        if score > best_score:
            best_score = score
            best_set = s2

    # Final check
    if best_set is None or len(best_set) != 32:
        best_set = set(range(1, 33))
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
