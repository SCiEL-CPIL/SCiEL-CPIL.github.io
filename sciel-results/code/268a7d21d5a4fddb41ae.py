import time
import random
import sys

def _check_sidon(els):
    """Check if set is Sidon."""
    if len(els) < 2:
        return True
    sums = set()
    for i in range(len(els)):
        for j in range(i, len(els)):
            s = els[i] + els[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _get_diffs(els):
    """Get set of nonzero differences."""
    diffs = set()
    for i in range(len(els)):
        for j in range(len(els)):
            if i != j:
                diffs.add(els[i] - els[j])
    return diffs

def _construct_A(target_size, n=256, rng=None):
    """Construct a random Sidon set of target_size."""
    if rng is None:
        rng = random.Random()
    
    A = []
    # Heuristic: Start with random elements, prefer larger gaps?
    # Just random shuffle and greedy add
    candidates = list(range(1, n + 1))
    rng.shuffle(candidates)
    
    for x in candidates:
        if len(A) == target_size:
            break
        # Check if x can be added to A
        valid = True
        for a in A:
            # Check sums
            s = a + x
            # We need to check against all existing sums
            # For efficiency, we can just re-check the whole set if small, 
            # or maintain a sum set.
            # Since target size is ~20, re-checking is O(k^2) which is small.
            pass
        
        # Simpler: Just check Sidon property of A + [x]
        new_A = A + [x]
        if _check_sidon(new_A):
            A.append(x)
            
    return A if len(A) == target_size else None

def _construct_B_given_A(A, target_size, n=256, rng=None):
    """Construct B given A, such that B is Sidon and D_B disjoint from D_A."""
    if rng is None:
        rng = random.Random()
        
    D_A = _get_diffs(A)
    B = []
    D_B = set()
    
    candidates = list(range(1, n + 1))
    rng.shuffle(candidates)
    
    # We might need backtracking if greedy fails.
    # Let's try a simple greedy first, then backtrack if needed.
    # Given the size (13), greedy might work with enough random restarts.
    
    def backtrack(B, D_B, idx):
        if len(B) == target_size:
            return True
        if idx >= len(candidates):
            return False
            
        x = candidates[idx]
        
        # Option 1: Skip x
        if backtrack(B, D_B, idx + 1):
            return True
            
        # Option 2: Add x
        # Check if x can be added to B
        valid = True
        new_diffs = set()
        for b in B:
            d1 = x - b
            d2 = b - x
            # Check against D_A
            if d1 in D_A or d2 in D_A:
                valid = False
                break
            # Check against existing D_B
            if d1 in D_B or d2 in D_B:
                valid = False
                break
            new_diffs.add(d1)
            new_diffs.add(d2)
            
        if valid:
            # Check Sidon property for B + [x]
            # This is equivalent to checking if new_diffs are unique?
            # No, Sidon is about sums.
            # But if differences are unique, sums are unique.
            # Wait, Sidon implies unique differences.
            # Does unique differences imply Sidon?
            # If $a+b = c+d$, then $a-c = d-b$.
            # If differences are unique, $a-c = d-b$ implies $a=c$ and $d=b$ or $a=d$ and $c=b$.
            # Since we consider distinct pairs for differences, this holds.
            # So checking unique differences is sufficient for Sidon property.
            
            # However, we must also ensure that the new element doesn't create a sum collision
            # that doesn't manifest as a difference collision?
            # No, sum collision $a+b=c+d \iff a-c=d-b$.
            # So checking differences is equivalent.
            
            # Add x to B
            B.append(x)
            D_B.update(new_diffs)
            
            if backtrack(B, D_B, idx + 1):
                return True
                
            # Backtrack
            B.pop()
            D_B.difference_update(new_diffs)
            
        return False

    # Run backtracking with a time limit or just let it run
    # Since target size is small, it should be fast.
    success = backtrack(B, D_B, 0)
    
    if success:
        return B
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 80
    n = 256
    
    best_A = [1]
    best_B = [2]
    best_score = 2
    
    rng = random.Random()
    
    # Target pairs that sum to 32
    targets = [
        (19, 13),
        (18, 14),
        (17, 15),
        (16, 16),
        (20, 12), # 32
        (21, 11), # 32
        (22, 10), # 32
        (23, 9),  # 32
        (24, 8),  # 32
        (25, 7),  # 32
        (26, 6),  # 32
        (27, 5),  # 32
        (28, 4),  # 32
        (29, 3),  # 32
        (30, 2),  # 32
        (31, 1),  # 32
        (32, 1)   # 33? No, 32+1=33. But binom(32,2) is huge.
    ]
    
    # Filter targets by feasibility (binom sum <= 255)
    feasible_targets = []
    for ka, kb in targets:
        if ka < 1 or kb < 1: continue
        if ka > 64 or kb > 64: continue
        diff_count = (ka*(ka-1) + kb*(kb-1)) // 2
        if diff_count <= 255:
            feasible_targets.append((ka, kb))
    
    # Shuffle targets to try different sizes
    rng.shuffle(feasible_targets)
    
    for ka, kb in feasible_targets:
        if time.time() > deadline:
            break
            
        attempts = 0
        while time.time() < deadline and attempts < 50:
            attempts += 1
            
            A = _construct_A(ka, n, rng)
            if A is None:
                continue
                
            B = _construct_B_given_A(A, kb, n, rng)
            if B is None:
                continue
                
            # Verify
            if _check_sidon(A) and _check_sidon(B):
                D_A = _get_diffs(A)
                D_B = _get_diffs(B)
                if not (D_A & D_B):
                    score = len(A) + len(B)
                    if score > best_score:
                        best_score = score
                        best_A = A
                        best_B = B
                        # If we hit 32, we can stop early? 
                        # Or keep going to see if we find 33?
                        # Max is likely 32.
                        if score >= 32:
                            # Try a few more to see if 33 is possible? 
                            # But 33 requires binom sum > 255?
                            # 17+16=33. Binom(17,2)=136, Binom(16,2)=120. Sum=256 > 255.
                            # So 33 is impossible.
                            break
        
    return [best_A, best_B]
