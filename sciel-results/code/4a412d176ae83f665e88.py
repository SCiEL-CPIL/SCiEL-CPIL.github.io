import time
import random
import sys
from itertools import combinations

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    time_limit = 85.0  # Safety margin below 90s

    # Helper to compute all consecutive block sums for a sequence
    def get_all_sums(seq):
        n = len(seq)
        sums = set()
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                sums.add(s)
        return sums

    # Helper to check if adding x to seq keeps all consecutive block sums distinct
    # Only need to check new sums that end at x
    def can_add(seq, x):
        # New sums are: x, seq[-1]+x, seq[-2]+seq[-1]+x, ..., sum(seq)+x
        total = x
        if total in _current_sums:
            return False
        for a in reversed(seq):
            total += a
            if total in _current_sums:
                return False
        return True

    # Global state for fast checking
    _current_sums = set()

    # Greedy construction with different strategies
    def greedy_construct(order):
        """Build a sequence greedily given an order of candidate values."""
        seq = []
        sums = set()
        for x in order:
            if x <= (seq[-1] if seq else 0):
                continue
            # Check if x can be added
            total = x
            conflict = False
            if total in sums:
                conflict = True
            else:
                for a in reversed(seq):
                    total += a
                    if total in sums:
                        conflict = True
                        break
            if not conflict:
                seq.append(x)
                # Update sums
                total = x
                sums.add(total)
                for a in reversed(seq[:-1]):
                    total += a
                    sums.add(total)
        return seq

    # Multiple greedy starts
    best_seq = []

    # Ascending greedy
    seq1 = greedy_construct(range(1, 257))
    if len(seq1) > len(best_seq):
        best_seq = seq1

    # Descending-ish greedy (try larger numbers first but maintain increasing order)
    # Actually we need strictly increasing, so we pick from high to low but sequence must increase
    # Let's try: start with a large number, then add larger ones
    # Better: try different starting points
    for start_val in [1, 2, 3, 5, 8, 10, 16, 32, 64, 128]:
        seq = greedy_construct(range(start_val, 257))
        if len(seq) > len(best_seq):
            best_seq = seq

    # Randomized greedy restarts
    random.seed(42)
    for _ in range(50):
        # Create a random permutation of candidates
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        # But we need increasing order in the result, so we sort candidates
        # and then use a random subset/order strategy
        # Actually, let's try: pick a random threshold and only consider values above it
        # Or: randomize the order but enforce increasing constraint
        # Better approach: random greedy with random tie-breaking
        seq = []
        sums = set()
        remaining = list(range(1, 257))
        random.shuffle(remaining)
        for x in remaining:
            if seq and x <= seq[-1]:
                continue
            total = x
            conflict = False
            if total in sums:
                conflict = True
            else:
                for a in reversed(seq):
                    total += a
                    if total in sums:
                        conflict = True
                        break
            if not conflict:
                seq.append(x)
                total = x
                sums.add(total)
                for a in reversed(seq[:-1]):
                    total += a
                    sums.add(total)
        if len(seq) > len(best_seq):
            best_seq = seq

    # Local search: try to insert elements into the best sequence found
    def local_search_insertion(seq):
        """Try to insert additional elements into seq to make it longer."""
        seq = seq[:]
        sums = get_all_sums(seq)
        improved = True
        while improved:
            improved = False
            # Try inserting each candidate in [1, 256] at each position
            for pos in range(len(seq) + 1):
                min_val = seq[pos - 1] + 1 if pos > 0 else 1
                max_val = seq[pos] - 1 if pos < len(seq) else 256
                for x in range(min_val, max_val + 1):
                    # Check if inserting x at pos keeps all sums distinct
                    new_seq = seq[:pos] + [x] + seq[pos:]
                    new_sums = get_all_sums(new_seq)
                    if len(new_sums) == len(new_seq) * (len(new_seq) + 1) // 2:
                        seq = new_seq
                        sums = new_sums
                        improved = True
                        break
                if improved:
                    break
        return seq

    # Apply local search to best_seq
    best_seq = local_search_insertion(best_seq)

    # Beam search for improvement
    def beam_search(initial_seq, beam_width=5, max_depth=10):
        """Beam search to find longer sequences."""
        beams = [initial_seq[:]]
        best = initial_seq[:]
        
        for depth in range(max_depth):
            if time.time() - start_time > time_limit:
                break
            new_beams = []
            for seq in beams:
                if len(seq) > len(best):
                    best = seq[:]
                
                # Try inserting one element
                sums = get_all_sums(seq)
                candidates_found = []
                for pos in range(len(seq) + 1):
                    min_val = seq[pos - 1] + 1 if pos > 0 else 1
                    max_val = seq[pos] - 1 if pos < len(seq) else 256
                    for x in range(min_val, max_val + 1):
                        new_seq = seq[:pos] + [x] + seq[pos:]
                        new_sums = get_all_sums(new_seq)
                        if len(new_sums) == len(new_seq) * (len(new_seq) + 1) // 2:
                            candidates_found.append(new_seq)
                            if len(candidates_found) >= beam_width * 2:
                                break
                    if len(candidates_found) >= beam_width * 2:
                        break
                
                # Keep top beam_width candidates by length
                candidates_found.sort(key=len, reverse=True)
                new_beams.extend(candidates_found[:beam_width])
            
            if not new_beams:
                break
            # Deduplicate and keep top beam_width
            unique_beams = []
            seen = set()
            for b in new_beams:
                key = tuple(b)
                if key not in seen:
                    seen.add(key)
                    unique_beams.append(b)
            unique_beams.sort(key=len, reverse=True)
            beams = unique_beams[:beam_width]
            
            if time.time() - start_time > time_limit:
                break
        
        return best

    best_seq = beam_search(best_seq, beam_width=3, max_depth=5)

    # Final local search pass
    best_seq = local_search_insertion(best_seq)

    # Verify the result
    final_sums = get_all_sums(best_seq)
    expected_count = len(best_seq) * (len(best_seq) + 1) // 2
    if len(final_sums) != expected_count:
        # Fallback to a known valid sequence
        best_seq = greedy_construct(range(1, 257))

    return best_seq
