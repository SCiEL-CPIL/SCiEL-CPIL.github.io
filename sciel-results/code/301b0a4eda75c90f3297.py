"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill

    def time_left():
        return deadline - time.time()

    def greedy_extend(seq, sums, start_val):
        """Greedily extend the sequence starting from start_val."""
        if not seq:
            start_val = 1
        last = seq[-1] if seq else 0
        for x in range(max(1, last + 1), 257):
            if x <= last:
                continue
            tails, total = [x], x
            for a in reversed(seq):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                seq.append(x)
                sums.update(tails)
        return seq, sums

    # First, get the greedy baseline
    base_seq, base_sums = greedy_extend([], set(), 1)
    best_seq = list(base_seq)
    best_len = len(best_seq)

    # Try different starting values to see if they yield longer sequences
    for start in range(1, 20):
        if time_left() < 5.0:
            break
        seq, sums = greedy_extend([], set(), start)
        if len(seq) > best_len:
            best_seq = list(seq)
            best_len = len(seq)

    # Beam search with diverse starting points
    def beam_search(start_val, beam_width=50, max_depth=256):
        """Beam search to find longer sequences."""
        if time_left() < 10.0:
            return []
        
        # Each beam entry: (sequence, set of sums, last_value)
        beams = []
        
        # Initialize with some diverse starting values
        for s in range(start_val, min(start_val + 15, 257)):
            seq = [s]
            sums = {s}
            beams.append((seq, sums))
        
        best_found = []
        
        while beams and time_left() > 15.0:
            # Sort beams by length (descending), then by last value (ascending) for diversity
            beams.sort(key=lambda b: (-len(b[0]), b[0][-1]))
            
            # Keep top beam_width
            if len(beams) > beam_width:
                beams = beams[:beam_width]
            
            new_beams = []
            for seq, sums in beams:
                last = seq[-1]
                # Try extending this sequence
                for x in range(last + 1, min(last + 30, 257)):
                    if time_left() < 15.0:
                        break
                    tails, total = [x], x
                    for a in reversed(seq):
                        total += a
                        tails.append(total)
                    if not sums.intersection(tails):
                        new_seq = seq + [x]
                        new_sums = sums.union(tails)
                        new_beams.append((new_seq, new_sums))
                        if len(new_seq) > len(best_found):
                            best_found = new_seq
            
            if not new_beams:
                break
            
            # Prune: keep diverse beams
            # Sort by length desc, then last value asc
            new_beams.sort(key=lambda b: (-len(b[0]), b[0][-1]))
            
            # Keep top beam_width, but ensure diversity in last values
            seen_last = set()
            diverse_beams = []
            for b in new_beams:
                if len(diverse_beams) >= beam_width:
                    break
                last_val = b[0][-1]
                # Allow some duplicates but prefer diversity
                if last_val not in seen_last or len(diverse_beams) < beam_width // 2:
                    diverse_beams.append(b)
                    seen_last.add(last_val)
            
            beams = diverse_beams if diverse_beams else new_beams[:beam_width]
        
        return best_found

    # Try beam search from a few different starting points
    for start in [1, 2, 3, 5, 8]:
        if time_left() < 20.0:
            break
        result = beam_search(start, beam_width=30)
        if len(result) > best_len:
            best_seq = result
            best_len = len(result)

    # Local improvement: try to insert values into the best sequence
    if time_left() > 30.0:
        improved = list(best_seq)
        improved_sums = set()
        for a in improved:
            improved_sums.add(a)
        for i in range(len(improved)):
            for j in range(i, len(improved)):
                s = sum(improved[i:j+1])
                improved_sums.add(s)
        
        # Try inserting new values
        for pos in range(len(improved) + 1):
            if time_left() < 10.0:
                break
            last_before = improved[pos-1] if pos > 0 else 0
            next_after = improved[pos] if pos < len(improved) else 257
            
            for x in range(last_before + 1, next_after):
                if x <= last_before or x >= next_after:
                    continue
                # Check if inserting x at pos maintains distinct sums
                new_seq = improved[:pos] + [x] + improved[pos:]
                new_sums = set()
                for i in range(len(new_seq)):
                    total = 0
                    for j in range(i, len(new_seq)):
                        total += new_seq[j]
                        if total in new_sums:
                            break
                        new_sums.add(total)
                    else:
                        continue
                    break
                else:
                    if len(new_seq) > best_len:
                        best_seq = new_seq
                        best_len = len(best_seq)
                        improved = new_seq
                        improved_sums = new_sums

    # Final greedy polish on the best sequence
    if time_left() > 15.0:
        final_seq, final_sums = greedy_extend(list(best_seq), set(), 1)
        # Rebuild sums properly
        final_sums = set()
        for i in range(len(final_seq)):
            total = 0
            for j in range(i, len(final_seq)):
                total += final_seq[j]
                final_sums.add(total)
        if len(final_seq) > best_len:
            best_seq = final_seq
            best_len = len(best_seq)

    # Verify the result
    if time_left() > 5.0:
        verify_sums = set()
        valid = True
        for i in range(len(best_seq)):
            total = 0
            for j in range(i, len(best_seq)):
                total += best_seq[j]
                if total in verify_sums:
                    valid = False
                    break
                verify_sums.add(total)
            if not valid:
                break
        if not valid:
            # Fallback to greedy
            best_seq, _ = greedy_extend([], set(), 1)

    # Ensure strictly increasing and in [1, 256]
    best_seq = [x for x in best_seq if 1 <= x <= 256]
    # Remove duplicates and ensure strictly increasing
    filtered = []
    for x in best_seq:
        if not filtered or x > filtered[-1]:
            filtered.append(x)
    best_seq = filtered

    return best_seq
# EVOLVE-BLOCK-END
