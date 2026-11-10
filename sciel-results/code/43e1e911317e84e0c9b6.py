import numpy as np
import time
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    # Generate all APs within [0, n)
    aps = []
    for d in range(1, n):
        for a in range(d):
            seq = list(range(a, n, d))
            if len(seq) >= 2:
                aps.append(seq)
    
    m = len(aps)
    
    # Precompute incidence matrix
    A = np.zeros((n, m), dtype=np.int8)
    for ap_id, ap in enumerate(aps):
        for idx in ap:
            A[idx, ap_id] = 1
    
    # Precompute which APs each index belongs to (for fast flip updates)
    index_to_ap_ids = [np.nonzero(A[i, :])[0] for i in range(n)]
    
    # Legendre symbol modulo 257 as initialization
    # Legendre symbol (a/p) = 1 if a is QR mod p, -1 if QNR, 0 if a≡0 mod p
    p = 257  # prime
    def legendre(a, p):
        if a % p == 0:
            return 0
        return pow(a, (p-1)//2, p) % p  # returns 1 or p-1=256
    
    # Build initial signs using Legendre symbol
    signs = np.zeros(n, dtype=np.int8)
    for i in range(n):
        ls = legendre(i+1, p)  # use i+1 to avoid 0
        if ls == 1:
            signs[i] = 1
        elif ls == 256:  # p-1
            signs[i] = -1
        else:
            signs[i] = 1  # shouldn't happen for i+1 in 1..256
    
    # Compute AP sums
    ap_sums = A.T @ signs
    current_max = np.max(np.abs(ap_sums))
    best_signs = signs.copy()
    best_max = current_max
    
    # Local search with simulated annealing
    def evaluate_flip(i, new_sign):
        """Evaluate the change in max AP sum if we flip index i to new_sign."""
        idxs = index_to_ap_ids[i]
        delta = 2 * new_sign  # change from -new_sign to new_sign
        old_sums = ap_sums[idxs]
        new_sums = old_sums + delta
        # Max of |new_sums| for affected APs, and max of |ap_sums| for unaffected
        local_max = np.max(np.abs(new_sums))
        # For unaffected APs, the max is current_max (unchanged)
        # But we need the global max after flip
        # The global max is max(local_max, max of unaffected)
        # Since we only changed affected APs, the global max is:
        # max(max of affected new values, max of unaffected old values)
        # The max of unaffected old values <= current_max
        # So global max after flip = max(local_max, current_max) if no unaffected AP exceeds current_max
        # Actually, we need to be more careful. Let's just compute it properly.
        # For efficiency, note that only affected APs change.
        # The new global max = max(max(|new_sums|), max(|ap_sums[unaffected]|))
        # max(|ap_sums[unaffected]|) <= current_max
        # So new_global_max = max(local_max, current_max)
        # Wait, that's not quite right either. The unaffected APs' max could be less than current_max.
        # Let's just compute it directly for correctness.
        return local_max
    
    # Better: just compute the new max properly
    def compute_new_max_after_flip(i, new_sign):
        idxs = index_to_ap_ids[i]
        delta = 2 * new_sign
        # Temporarily update
        old_vals = ap_sums[idxs].copy()
        ap_sums[idxs] += delta
        new_max = np.max(np.abs(ap_sums))
        ap_sums[idxs] = old_vals
        return new_max
    
    no_improve_count = 0
    max_no_improve = 30
    temperature = 2.0
    cooling = 0.95
    
    while time.time() < deadline - 5:
        improved = False
        indices = list(range(n))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline - 3:
                break
            
            old_sign = signs[i]
            new_sign = -old_sign
            
            # Compute new max if we flip
            idxs = index_to_ap_ids[i]
            delta = 2 * new_sign
            old_vals = ap_sums[idxs].copy()
            ap_sums[idxs] += delta
            new_max = np.max(np.abs(ap_sums))
            
            # Acceptance criterion
            if new_max < current_max:
                # Always accept improvement
                current_max = new_max
                signs[i] = new_sign
                improved = True
            else:
                # Simulated annealing: accept worse with some probability
                delta_max = new_max - current_max
                if delta_max > 0 and random.random() < np.exp(-delta_max / temperature):
                    current_max = new_max
                    signs[i] = new_sign
                    improved = True
                else:
                    # Revert
                    ap_sums[idxs] = old_vals
                    # signs[i] stays as old_sign
            
        if improved:
            no_improve_count = 0
            if current_max < best_max:
                best_max = current_max
                best_signs = signs.copy()
        else:
            no_improve_count += 1
            temperature *= cooling
            if no_improve_count > max_no_improve:
                # Restart from best with perturbation
                if best_max > 3:
                    signs = best_signs.copy()
                    # Perturb a few random positions
                    num_perturb = random.randint(3, 8)
                    for _ in range(num_perturb):
                        idx = random.randint(0, n-1)
                        signs[idx] = -signs[idx]
                    ap_sums = A.T @ signs
                    current_max = np.max(np.abs(ap_sums))
                    temperature = 1.5
                    no_improve_count = 0
                else:
                    break
    
    return best_signs.tolist()
