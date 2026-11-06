"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

def construct():
    deadline = time.time() + 85  # 85s safety margin (limit is 90s)
    
    def compute_sums(seq):
        """Compute all consecutive block sums for a sequence."""
        sums = set()
        n = len(seq)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                sums.add(s)
        return sums
    
    def is_valid(seq):
        """Check if sequence has distinct consecutive block sums."""
        sums = set()
        n = len(seq)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    def greedy_standard():
        """Standard greedy: pick smallest available x at each step."""
        values, sums = [], set()
        for x in range(1, 257):
            tails, total = [x], x
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        return values, sums
    
    def systematic_search(max_depth, time_limit):
        """
        Systematic DFS search with priority on smaller candidates.
        Uses iterative deepening to find longest valid sequence.
        """
        best = []
        best_len = 0
        start_time = time.time()
        
        def dfs(seq, sums, depth, max_d):
            nonlocal best, best_len, start_time
            if time.time() > time_limit:
                return False
            
            if len(seq) > best_len:
                best = list(seq)
                best_len = len(seq)
            
            if depth == max_d:
                return True
            
            last = seq[-1] if seq else 0
            
            # Try candidates from last+1 upward (prioritize smaller)
            for x in range(last + 1, 257):
                if time.time() > time_limit:
                    return False
                
                # Compute new tails (suffix sums ending at x)
                tails = []
                total = x
                tails.append(total)
                for a in reversed(seq):
                    total += a
                    tails.append(total)
                
                # Check if any tail conflicts
                if any(t in sums for t in tails):
                    continue
                
                # Valid - recurse
                seq.append(x)
                sums.update(tails)
                dfs(seq, sums, depth + 1, max_d)
                sums.difference_update(tails)
                seq.pop()
                
                if time.time() > time_limit:
                    return False
        
        # Start with greedy baseline
        base_seq, base_sums = greedy_standard()
        best = list(base_seq)
        best_len = len(base_seq)
        
        # Try to extend beyond greedy baseline using systematic search
        # Use iterative deepening: try to find sequences of length base_len+1, +2, etc.
        current_limit = time_limit
        for target_len in range(len(base_seq) + 1, len(base_seq) + 20):
            if time.time() > current_limit:
                break
            
            # Time budget for this depth
            remaining = current_limit - time.time()
            if remaining < 1:
                break
            
            dfs(list(base_seq), set(base_sums), len(base_seq), target_len)
            
            if len(best) >= target_len:
                # Found a sequence of this length, try to extend further
                # Update base for next iteration
                base_seq = list(best)
                base_sums = compute_sums(base_seq)
            else:
                # Couldn't find sequence of this length, break
                break
        
        return best
    
    def randomized_systematic(seed, time_limit):
        """
        Randomized systematic search: use randomness to explore different branches.
        """
        rng = random.Random(seed)
        best = []
        best_len = 0
        start_time = time.time()
        
        def dfs(seq, sums, depth):
            nonlocal best, best_len
            if time.time() > time_limit:
                return
            
            if len(seq) > best_len:
                best = list(seq)
                best_len = len(seq)
            
            last = seq[-1] if seq else 0
            
            # Generate candidate list and shuffle for randomness
            candidates = list(range(last + 1, 257))
            rng.shuffle(candidates)
            
            for x in candidates:
                if time.time() > time_limit:
                    return
                
                # Compute new tails
                tails = []
                total = x
                tails.append(total)
                for a in reversed(seq):
                    total += a
                    tails.append(total)
                
                # Check if any tail conflicts
                if any(t in sums for t in tails):
                    continue
                
                # Valid - recurse
                seq.append(x)
                sums.update(tails)
                dfs(seq, sums, depth + 1)
                sums.difference_update(tails)
                seq.pop()
                
                if time.time() > time_limit:
                    return
        
        dfs([], set(), 0)
        return best
    
    def beam_search(beam_width, time_limit):
        """
        Beam search: keep top beam_width states at each depth.
        State = (sequence, sums, last_element)
        Priority: prefer smaller last elements (more room to grow)
        """
        if time.time() > time_limit:
            return []
        
        # Initial state: empty sequence
        states = [( [], set(), 0)]
        best = []
        best_len = 0
        
        current_depth = 0
        while time.time() < time_limit:
            next_states = []
            for seq, sums, last in states:
                if time.time() > time_limit:
                    break
                
                if len(seq) > best_len:
                    best = list(seq)
                    best_len = len(seq)
                
                # Try extending with each valid candidate
                for x in range(last + 1, 257):
                    if time.time() > time_limit:
                        break
                    
                    # Compute new tails
                    tails = []
                    total = x
                    tails.append(total)
                    for a in reversed(seq):
                        total += a
                        tails.append(total)
                    
                    # Check if any tail conflicts
                    if any(t in sums for t in tails):
                        continue
                    
                    # Valid
                    new_sums = sums | set(tails)
                    next_states.append((seq + [x], new_sums, x))
            
            if not next_states:
                break
            
            # Keep only top beam_width states
            # Sort by last element (prefer smaller for more room) and sequence length
            next_states.sort(key=lambda s: (s[2], -len(s[0])))
            states = next_states[:beam_width]
            current_depth += 1
            
            # If we've exhausted the search space or reached time limit, stop
            if len(states) < beam_width or current_depth > 30:
                break
        
        return best
    
    # Get baseline
    base_seq, base_sums = greedy_standard()
    best = list(base_seq)
    
    # Time allocation:
    # 1. Systematic search from baseline: 40s
    # 2. Randomized systematic: 20s  
    # 3. Beam search: 15s
    # 4. Final polish: 5s
    
    # Phase 1: Systematic search
    t1 = time.time()
    sys_best = systematic_search(len(base_seq), t1 + 40)
    if len(sys_best) > len(best):
        best = sys_best
    
    # Phase 2: Randomized systematic search
    t2 = time.time()
    if t2 < deadline - 25:
        for seed in range(50):
            if time.time() > deadline - 25:
                break
            rand_best = randomized_systematic(seed, deadline - 25)
            if len(rand_best) > len(best):
                best = rand_best
    
    # Phase 3: Beam search
    t3 = time.time()
    if t3 < deadline - 15:
        beam_best = beam_search(100, deadline - 15)
        if len(beam_best) > len(best):
            best = beam_best
    
    # Phase 4: Final polish - try to extend best with greedy
    if time.time() < deadline - 5:
        last = best[-1] if best else 0
        sums = compute_sums(best)
        for x in range(last + 1, 257):
            if time.time() > deadline - 5:
                break
            tails, total = [x], x
            for a in reversed(best):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                best.append(x)
                sums.update(tails)
    
    return best
# EVOLVE-BLOCK-END
