import time
import random
import numpy as np

def construct():
    deadline = time.time() + 85.0
    rng = random.Random(12345)
    
    n = 128
    target_lo = 128
    target_hi = 256
    target_range = target_hi - target_lo + 1  # 129 values
    
    def compute_multiplicities(mask):
        """Use numpy convolution to compute pair-sum multiplicities.
        mask is a boolean array of length n+1 (indices 0..n).
        Returns array of length 2n+1 with counts for sums 0..2n."""
        # Autocorrelation via convolution: convolve mask with reversed mask
        # For unordered pairs with equal summands allowed:
        # count[s] = sum_{a+b=s} [a in A][b in A] but we need unordered
        # Actually, for the sum s, the number of unordered pairs (a,b) with a<=b, a+b=s, a,b in A
        # We can compute: for each a in A, for each b in A with b>=a, add to count[a+b]
        # But numpy approach: convolve the indicator with itself gives ordered pairs.
        # For unordered: count_unordered[s] = (ordered[s] + (s even and s/2 in A)) / 2
        # Wait, that's not quite right either. Let me think...
        # ordered[s] = sum_{a} [a in A][s-a in A] = number of ordered pairs (a,b) with a+b=s
        # unordered[s] = number of pairs {a,b} with a<=b, a+b=s, a,b in A
        # = (ordered[s] + diag[s]) / 2 where diag[s] = 1 if s is even and s/2 in A else 0
        # Actually: ordered[s] = 2*(#pairs with a<b) + (#pairs with a=b)
        # unordered[s] = (#pairs with a<b) + (#pairs with a=b)
        # So unordered[s] = (ordered[s] + (#pairs with a=b)) / 2
        # #pairs with a=b for sum s: 1 if s even and s/2 in A, else 0
        conv = np.convolve(mask.astype(np.int32), mask.astype(np.int32), mode='full')
        # conv[k] = sum_{i+j=k} mask[i]*mask[j] = ordered count for sum k
        # For sum s (0 <= s <= 2n):
        diag = np.zeros(2*n+1, dtype=np.int32)
        for s in range(0, 2*n+1, 2):
            half = s // 2
            if 0 <= half <= n:
                diag[s] = mask[half]
        unordered = (conv + diag) // 2
        return unordered
    
    def evaluate(mask):
        """Return (peak_multiplicity, is_valid)."""
        mult = compute_multiplicities(mask)
        # Check coverage: every t in [128, 256] must have mult[t] > 0
        target_slice = mult[target_lo:target_hi+1]
        if np.any(target_slice == 0):
            return float('inf'), False
        peak = int(np.max(target_slice))
        return peak, True
    
    def fast_check_coverage(mask):
        """Quick check if all targets are covered. Returns True if valid coverage."""
        mult = compute_multiplicities(mask)
        target_slice = mult[target_lo:target_hi+1]
        return bool(np.all(target_slice > 0))
    
    def get_peak(mask):
        """Get peak multiplicity assuming coverage is already verified."""
        mult = compute_multiplicities(mask)
        return int(np.max(mult[target_lo:target_hi+1]))
    
    def local_search(mask, max_steps=2000):
        """Local search: try single moves (add, remove, swap) to reduce peak."""
        best_peak, valid = evaluate(mask)
        if not valid:
            return mask, best_peak
        current_peak = best_peak
        mask = mask.copy()
        current = mask.copy()
        
        for step in range(max_steps):
            if time.time() > deadline - 2.0:
                break
            
            improved = False
            # Try removals
            ones = np.where(current)[0]
            rng.shuffle(ones.tolist())
            for idx in ones:
                if time.time() > deadline - 2.0:
                    break
                new_mask = current.copy()
                new_mask[idx] = 0
                if fast_check_coverage(new_mask):
                    p = get_peak(new_mask)
                    if p < current_peak:
                        current = new_mask
                        current_peak = p
                        improved = True
                        break
                elif np.all(compute_multiplicities(new_mask)[target_lo:target_hi+1] > 0):
                    p = get_peak(new_mask)
                    if p < current_peak:
                        current = new_mask
                        current_peak = p
                        improved = True
                        break
            
            if improved:
                continue
            
            # Try additions
            zeros = np.where(~current)[0]
            rng.shuffle(zeros.tolist())
            for idx in zeros:
                if time.time() > deadline - 2.0:
                    break
                new_mask = current.copy()
                new_mask[idx] = 1
                p = get_peak(new_mask)
                if p < current_peak:
                    current = new_mask
                    current_peak = p
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swaps
            ones = np.where(current)[0]
            zeros = np.where(~current)[0]
            if len(ones) > 0 and len(zeros) > 0:
                for _ in range(min(50, len(ones)*len(zeros))):
                    if time.time() > deadline - 2.0:
                        break
                    a = rng.choice(ones.tolist())
                    b = rng.choice(zeros.tolist())
                    new_mask = current.copy()
                    new_mask[a] = 0
                    new_mask[b] = 1
                    if fast_check_coverage(new_mask):
                        p = get_peak(new_mask)
                        if p < current_peak:
                            current = new_mask
                            current_peak = p
                            improved = True
                            break
            
            if not improved:
                # Try double moves: remove two, add one, or remove one, add two
                break
        
        return current, current_peak
    
    def generate_initial_mask(rng, strategy='random'):
        """Generate an initial mask."""
        mask = np.zeros(n+1, dtype=np.int8)
        if strategy == 'contiguous':
            # Contiguous block
            size = rng.randint(40, 80)
            start = rng.randint(0, n - size + 1)
            mask[start:start+size] = 1
        elif strategy == 'random':
            # Random subset
            size = rng.randint(50, 90)
            indices = rng.sample(range(n+1), size)
            mask[indices] = 1
        elif strategy == 'structured':
            # Include endpoints and some structure
            mask[0] = 1
            mask[n] = 1
            # Add some random
            size = rng.randint(30, 60)
            indices = rng.sample(range(1, n), size)
            mask[indices] = 1
        elif strategy == 'balanced':
            # Try to balance coverage
            # Include elements that help cover both low and high targets
            mask[0] = 1
            mask[n] = 1
            # Add elements around n/2
            for i in range(n//4, 3*n//4 + 1):
                if rng.random() < 0.7:
                    mask[i] = 1
            # Add some random extras
            for i in range(n+1):
                if mask[i] == 0 and rng.random() < 0.3:
                    mask[i] = 1
        return mask
    
    def repair(mask):
        """Try to repair coverage by adding elements."""
        mask = mask.copy()
        for _ in range(100):
            if time.time() > deadline - 2.0:
                break
            mult = compute_multiplicities(mask)
            uncovered = [t for t in range(target_lo, target_hi+1) if mult[t] == 0]
            if not uncovered:
                break
            # Find element that covers most uncovered targets
            best_elem = None
            best_cover = 0
            for elem in range(n+1):
                if mask[elem]:
                    continue
                # How many uncovered targets can this element help cover?
                # Adding elem creates new sums: elem + a for a in A
                cover = 0
                for t in uncovered:
                    other = t - elem
                    if 0 <= other <= n and mask[other]:
                        cover += 1
                if cover > best_cover:
                    best_cover = cover
                    best_elem = elem
            if best_elem is not None and best_cover > 0:
                mask[best_elem] = 1
            else:
                # Random add
                zeros = np.where(~mask)[0]
                if len(zeros) > 0:
                    mask[rng.choice(zeros)] = 1
                else:
                    break
        return mask
    
    # Phase 1: Try multiple initializations and local search
    best_mask = None
    best_peak = float('inf')
    
    strategies = ['contiguous', 'random', 'structured', 'balanced']
    
    for strat in strategies:
        for trial in range(5):
            if time.time() > deadline - 5.0:
                break
            
            mask = generate_initial_mask(rng, strat)
            
            # Ensure non-empty
            if not np.any(mask):
                mask[rng.randint(0, n)] = 1
            
            # Repair if needed
            if not fast_check_coverage(mask):
                mask = repair(mask)
                if not fast_check_coverage(mask):
                    continue
            
            # Local search
            mask, peak = local_search(mask, max_steps=3000)
            
            if peak < best_peak:
                best_peak = peak
                best_mask = mask.copy()
    
    # Phase 2: More intensive local search on best solution
    if best_mask is not None:
        for _ in range(3):
            if time.time() > deadline - 3.0:
                break
            best_mask, best_peak = local_search(best_mask, max_steps=5000)
    
    # Phase 3: Try perturbation + local search (simulated annealing style)
    if best_mask is not None:
        current_mask = best_mask.copy()
        current_peak = best_peak
        for _ in range(20):
            if time.time() > deadline - 3.0:
                break
            # Perturb: random swaps/adds/removes
            perturbed = current_mask.copy()
            num_changes = rng.randint(2, 6)
            for _ in range(num_changes):
                action = rng.random()
                if action < 0.4:
                    # Remove
                    ones = np.where(perturbed)[0]
                    if len(ones) > 3:
                        idx = rng.choice(ones.tolist())
                        perturbed[idx] = 0
                elif action < 0.7:
                    # Add
                    zeros = np.where(~perturbed)[0]
                    if len(zeros) > 0:
                        idx = rng.choice(zeros.tolist())
                        perturbed[idx] = 1
                else:
                    # Swap
                    ones = np.where(perturbed)[0]
                    zeros = np.where(~perturbed)[0]
                    if len(ones) > 0 and len(zeros) > 0:
                        a = rng.choice(ones.tolist())
                        b = rng.choice(zeros.tolist())
                        perturbed[a] = 0
                        perturbed[b] = 1
            
            # Repair
            if not fast_check_coverage(perturbed):
                perturbed = repair(perturbed)
                if not fast_check_coverage(perturbed):
                    continue
            
            # Local search
            perturbed, p = local_search(perturbed, max_steps=2000)
            
            if p < current_peak:
                current_peak = p
                current_mask = perturbed.copy()
                if p < best_peak:
                    best_peak = p
                    best_mask = current_mask.copy()
    
    # Final result
    if best_mask is None:
        # Fallback: use the seed solution
        mask = np.zeros(n+1, dtype=np.int8)
        mask[64:129] = 1
        return list(np.where(mask)[0])
    
    return list(np.where(best_mask)[0])
