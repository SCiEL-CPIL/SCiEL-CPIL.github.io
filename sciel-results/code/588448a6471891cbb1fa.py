import random
import time
import numpy as np

def construct():
    N = 256
    deadline = time.time() + 85  # Safety margin
    
    # Generate all APs
    # An AP is defined by start a, step d, length k
    # a, a+d, ..., a+(k-1)d <= N
    all_aps = []
    for d in range(1, N + 1):
        for a in range(1, N + 1):
            if a > N:
                break
            # Calculate max length
            # a + (k-1)d <= N => k <= (N-a)/d + 1
            max_k = (N - a) // d + 1
            # We consider APs of length >= 1
            # To reduce count, we might only consider "maximal" APs or all?
            # The problem says "ANY arithmetic progression".
            # This includes sub-APs.
            # However, the sum of a sub-AP is bounded by the sum of the maximal AP?
            # No, signs can cancel.
            # But typically, the worst discrepancy is found in longer APs.
            # For N=256, total APs is ~260k.
            # Storing all is okay.
            for k in range(1, max_k + 1):
                indices = list(range(a, a + k * d, d))
                all_aps.append(indices)
    
    # Optimize: Convert to numpy arrays for faster indexing?
    # np.sum(signs[indices]) is fast.
    # But creating 260k numpy arrays is slow.
    # Let's keep them as lists and use a helper.
    
    # Actually, 260k lists of varying lengths might be memory heavy but okay.
    # Let's try to be smarter.
    # We don't need ALL APs in the critical set initially.
    
    # Initial Critical Set:
    # Pick APs with small steps and large lengths.
    # d=1, length > 10
    # d=2, length > 10
    # etc.
    
    critical_aps = []
    # Add some random APs to start
    for _ in range(50):
        critical_aps.append(random.choice(all_aps))
    
    # Also add the "longest" APs for small d
    for d in range(1, 10):
        for a in range(1, N + 1, d): # Start at 1, d+1... no, just sample
            pass # Too complex to sample perfectly, just random is fine for init
    
    # Better Init:
    # Use the APs that are "dense"
    dense_aps = [ap for ap in all_aps if len(ap) > 20]
    if dense_aps:
        critical_aps.extend(random.sample(dense_aps, min(50, len(dense_aps))))
    
    # Deduplicate critical_aps (using tuple for hash)
    critical_aps = list(set(tuple(ap) for ap in critical_aps))
    # Convert back to lists? Or keep tuples?
    # np.sum works with tuples of indices? No, needs array or list.
    # Let's keep them as lists.
    critical_aps = [list(ap) for ap in critical_aps]
    
    # Initial Signs
    rng = random.Random(42)
    signs = [rng.choice((-1, 1)) for _ in range(N)]
    
    best_signs = signs.copy()
    best_score = float('inf')
    
    def eval_max(signs, ap_list):
        max_val = 0
        worst_ap = None
        for ap in ap_list:
            # Sum signs at indices in ap
            # Indices are 1-based in AP definition, 0-based in list
            # Wait, my AP generation used 1-based indices (a in 1..N)
            # But signs list is 0-based.
            # So index in signs is ap[i] - 1
            s = 0
            for idx in ap:
                s += signs[idx - 1]
            abs_s = abs(s)
            if abs_s > max_val:
                max_val = abs_s
                worst_ap = ap
        return max_val, worst_ap

    def eval_max_full(signs):
        return eval_max(signs, all_aps)

    iter_count = 0
    no_improve_count = 0
    
    while time.time() < deadline:
        iter_count += 1
        
        current_score, worst_ap = eval_max(signs, critical_aps)
        
        # Update best
        if current_score < best_score:
            best_score = current_score
            best_signs = signs.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        # Periodically check full set
        if iter_count % 10 == 0:
            full_score, full_worst_ap = eval_max_full(signs)
            if full_score > current_score:
                # Add the full worst ap to critical set
                if full_worst_ap not in critical_aps:
                    critical_aps.append(full_worst_ap)
                # Also, if full score is much higher, reset critical set to include more
                if full_score > current_score + 2:
                    # Add some random APs from the full set
                    for _ in range(10):
                        critical_aps.append(random.choice(all_aps))
                    # Deduplicate
                    critical_aps = list(set(tuple(ap) for ap in critical_aps))
                    critical_aps = [list(ap) for ap in critical_aps]
                    
        # Local Search: Try to flip a sign to reduce current_score
        # We focus on the worst_ap found
        if worst_ap:
            # Try flipping each element in the worst AP
            indices_to_try = [idx - 1 for idx in worst_ap]
            # Shuffle to avoid bias
            random.shuffle(indices_to_try)
            
            improved = False
            for i in indices_to_try:
                signs[i] *= -1
                new_score, _ = eval_max(signs, critical_aps)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    break
                else:
                    signs[i] *= -1 # Revert
                    
            if not improved:
                # Try flipping a random element
                i = rng.randint(0, N - 1)
                signs[i] *= -1
                new_score, _ = eval_max(signs, critical_aps)
                if new_score > current_score:
                    signs[i] *= -1 # Revert if it got worse
                    # If we are stuck, maybe accept a worse move (Simulated Annealing)?
                    # For now, just revert.
                else:
                    current_score = new_score

    # Final verification and polish
    # Do a few more local steps on the best_signs
    signs = best_signs.copy()
    for _ in range(100):
        if time.time() > deadline - 2:
            break
        current_score, worst_ap = eval_max(signs, critical_aps)
        if worst_ap:
            indices_to_try = [idx - 1 for idx in worst_ap]
            random.shuffle(indices_to_try)
            for i in indices_to_try:
                signs[i] *= -1
                new_score, _ = eval_max(signs, critical_aps)
                if new_score < current_score:
                    current_score = new_score
                    break
                else:
                    signs[i] *= -1
                    
    # One last full check to see if we can improve critical set and do one more pass
    full_score, full_worst_ap = eval_max_full(signs)
    if full_score > current_score:
        critical_aps.append(full_worst_ap)
        # Quick local search on this new worst ap
        indices_to_try = [idx - 1 for idx in full_worst_ap]
        random.shuffle(indices_to_try)
        for i in indices_to_try:
            signs[i] *= -1
            new_score, _ = eval_max(signs, critical_aps)
            if new_score < full_score:
                full_score = new_score
                break
            else:
                signs[i] *= -1

    return signs
