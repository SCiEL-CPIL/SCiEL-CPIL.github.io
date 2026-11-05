import numpy as np
import time

BASE = 51975
DIVISORS = [d for d in range(3, BASE + 1, 2) if BASE % d == 0]
# Sort divisors: larger moduli first (fewer residues to check), then by constraint priority
# A divisor m is more "constrained" if it has fewer divisors d of BASE such that d % m == 0
# (i.e., fewer other moduli can cover the same residues)
_divisor_constraint = {}
for m in DIVISORS:
    # Count how many other divisors are multiples of m
    cnt = sum(1 for d in DIVISORS if d % m == 0 and d != m)
    _divisor_constraint[m] = cnt

# Prioritize: high constraint (fewer covering alternatives) first, then larger m (cheaper to evaluate)
DIVISORS_SORTED = sorted(DIVISORS, key=lambda m: (-_divisor_constraint[m], -m))

def _compute_gain_fast(ref_count, m, r):
    """Fast gain computation: count uncovered residues in r::m."""
    return np.count_nonzero(ref_count[r::m] == 0)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    best_score = BASE
    best_classes = None
    
    # Precompute slice indices for each divisor to avoid repeated slicing
    # For each m, precompute the list of start indices r where we'd evaluate
    # Actually, we'll just use numpy slicing which is already efficient
    
    for restart in range(50):
        if time.time() > deadline:
            break
            
        # Initialize ref_count to 0
        ref_count = np.zeros(BASE, dtype=np.int8)
        
        # Assign random residues, but use constraint-priority initialization
        residues = {}
        for m in DIVISORS_SORTED:
            residues[m] = np.random.randint(0, m)
            ref_count[residues[m]::m] += 1
            
        # Calculate initial score
        score = int(np.sum(ref_count == 0))
        
        if score < best_score:
            best_score = score
            best_classes = [[m, residues[m]] for m in DIVISORS]
            if best_score == 0:
                return best_classes

        # Local Search (Hill Climbing) with constraint-priority ordering
        no_improve_rounds = 0
        max_no_improve = 3  # Allow a few rounds without improvement before restarting
        
        while time.time() < deadline:
            improved = False
            # Use constraint-priority order, with occasional shuffling for diversity
            if np.random.random() < 0.3:
                current_divisors = DIVISORS_SORTED.copy()
                np.random.shuffle(current_divisors)
            else:
                current_divisors = DIVISORS_SORTED
            
            for m in current_divisors:
                if time.time() > deadline:
                    break
                    
                r_curr = residues[m]
                
                # Remove current class
                ref_count[r_curr::m] -= 1
                
                # Calculate loss: how many became uncovered?
                loss = int(np.sum(ref_count[r_curr::m] == 0))
                
                # Find best new residue with incremental filtering
                best_r = r_curr
                best_gain = -1
                
                # For small m, check all residues. For large m, sample a subset first.
                if m <= 200:
                    candidates = range(m)
                else:
                    # Sample a subset of candidates, then refine
                    sample_size = min(m, 50)
                    candidates = np.random.choice(m, size=sample_size, replace=False)
                    # Also always include a few structured candidates
                    extra = [0, m//2, m//3]
                    for e in extra:
                        if e not in candidates:
                            candidates = np.append(candidates, e)
                    candidates = np.unique(candidates)
                
                for r in candidates:
                    r = int(r)
                    gain = _compute_gain_fast(ref_count, m, r)
                    if gain > best_gain:
                        best_gain = gain
                        best_r = r
                
                # If we sampled, do a second pass with more candidates if time allows
                if m > 200 and time.time() < deadline - 5:
                    # Expand search: check all residues if m is not too large
                    if m <= 1000:
                        for r in range(m):
                            gain = _compute_gain_fast(ref_count, m, r)
                            if gain > best_gain:
                                best_gain = gain
                                best_r = r
                
                # Apply best new class
                ref_count[best_r::m] += 1
                
                # Update score
                delta = loss - best_gain
                if delta < 0:
                    score += delta
                    improved = True
                    if score < best_score:
                        best_score = score
                        best_classes = [[m, residues[m]] for m in DIVISORS]
                        if best_score == 0:
                            return best_classes
                
                residues[m] = best_r
                
            if not improved:
                no_improve_rounds += 1
                if no_improve_rounds >= max_no_improve:
                    # Try random perturbations to escape local minimum
                    escaped = False
                    for _ in range(20):
                        if time.time() > deadline:
                            break
                        m = int(np.random.choice(DIVISORS))
                        r_old = residues[m]
                        r_new = np.random.randint(0, m)
                        if r_new == r_old:
                            continue
                        
                        ref_count[r_old::m] -= 1
                        loss = int(np.sum(ref_count[r_old::m] == 0))
                        ref_count[r_new::m] += 1
                        gain = int(np.sum(ref_count[r_new::m] == 0))
                        # Note: gain here is AFTER addition, so it's not the same as before
                        # We need to compute the actual score change
                        # Old uncovered count in the affected regions
                        # Simpler: just accept and recompute score locally
                        # But that's expensive. Let's just accept if it doesn't hurt too much.
                        # For now, just apply and break
                        score = int(np.sum(ref_count == 0))
                        residues[m] = r_new
                        escaped = True
                        break
                    
                    if not escaped:
                        break
                    no_improve_rounds = 0
            else:
                no_improve_rounds = 0

    return best_classes
