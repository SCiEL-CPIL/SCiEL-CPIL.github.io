"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 132  # safety margin before 150s kill
    
    base = 51975
    
    # Get all odd divisors of 51975 greater than 1
    divisors = []
    for d in range(3, base + 1, 2):
        if base % d == 0:
            divisors.append(d)
    
    # Sort by size descending - larger moduli first to cover unique residues
    divisors.sort(reverse=True)
    
    # Track coverage using numpy for speed
    covered = np.zeros(base, dtype=np.uint8)
    classes = []
    
    # Phase 1: Greedy - process moduli in decreasing order
    # For each modulus, pick the residue that covers the most uncovered residues
    for m in divisors:
        if time.time() > deadline - 5:
            break
        
        # Find best residue for this modulus using numpy
        best_r = 0
        best_count = -1
        
        if m <= 300:
            # Try all residues for small moduli
            for r in range(m):
                positions = np.arange(r, base, m)
                count = np.sum(covered[positions] == 0)
                if count > best_count:
                    best_count = count
                    best_r = r
        else:
            # For large moduli, try strategic candidates
            candidates = list(range(min(60, m)))
            # Add spread-out candidates
            for k in range(1, min(m, 25)):
                candidates.append(k * m // 25)
            candidates = list(set(c % m for c in candidates))
            
            for r in candidates:
                positions = np.arange(r, base, m)
                count = np.sum(covered[positions] == 0)
                if count > best_count:
                    best_count = count
                    best_r = r
            
            # Also try random samples
            rng = np.random.RandomState(len(classes) * 12345 + 67890)
            for _ in range(30):
                r = int(rng.randint(m))
                positions = np.arange(r, base, m)
                count = np.sum(covered[positions] == 0)
                if count > best_count:
                    best_count = count
                    best_r = r
        
        classes.append([m, best_r])
        
        # Update coverage using numpy
        positions = np.arange(best_r, base, m)
        covered[positions] = 1
    
    # Phase 2: Local search - try to improve by adjusting residues
    if time.time() < deadline - 20:
        improved = True
        max_iter = 80
        iteration = 0
        while improved and time.time() < deadline - 20 and iteration < max_iter:
            improved = False
            iteration += 1
            
            # Try improving each class
            for i in range(len(classes)):
                if time.time() > deadline - 20:
                    break
                
                m, r = classes[i]
                
                # Compute coverage without this class
                covered_temp = np.zeros(base, dtype=np.uint8)
                for j, (m2, r2) in enumerate(classes):
                    if j != i:
                        positions = np.arange(r2, base, m2)
                        covered_temp[positions] = 1
                
                base_uncovered = np.sum(covered_temp == 0)
                
                best_r2 = r
                best_uncovered = base_uncovered
                
                if m <= 300:
                    # Try all residues for small/moderate moduli
                    for r2 in range(m):
                        positions = np.arange(r2, base, m)
                        new_uncovered = base_uncovered - np.sum((covered_temp[positions] == 0).astype(np.uint8))
                        if new_uncovered < best_uncovered:
                            best_uncovered = new_uncovered
                            best_r2 = r2
                else:
                    # Sample residues for large moduli
                    rng = np.random.RandomState(i * 98765 + 43210 + iteration)
                    candidates = list(range(min(80, m)))
                    for k in range(1, min(m, 30)):
                        candidates.append(k * m // 30)
                    candidates = list(set(c % m for c in candidates))
                    for _ in range(60):
                        candidates.append(int(rng.randint(m)))
                    
                    for r2 in set(candidates):
                        positions = np.arange(r2, base, m)
                        new_uncovered = base_uncovered - np.sum((covered_temp[positions] == 0).astype(np.uint8))
                        if new_uncovered < best_uncovered:
                            best_uncovered = new_uncovered
                            best_r2 = r2
                
                if best_r2 != r:
                    classes[i] = [m, best_r2]
                    improved = True
    
    # Phase 3: Try removing classes that don't help
    if time.time() < deadline - 10:
        removed = True
        while removed and time.time() < deadline - 10:
            removed = False
            
            covered_current = np.zeros(base, dtype=np.uint8)
            for m, r in classes:
                positions = np.arange(r, base, m)
                covered_current[positions] = 1
            
            current_uncovered = np.sum(covered_current == 0)
            
            for i in range(len(classes) - 1, -1, -1):
                if time.time() > deadline - 10:
                    break
                
                # Check if removing this class increases uncovered
                covered_without = np.zeros(base, dtype=np.uint8)
                for j, (m, r) in enumerate(classes):
                    if j != i:
                        positions = np.arange(r, base, m)
                        covered_without[positions] = 1
                
                unc_without = np.sum(covered_without == 0)
                
                # If removing doesn't increase uncovered, remove it
                if unc_without <= current_uncovered:
                    classes.pop(i)
                    removed = True
                    current_uncovered = unc_without
    
    # Phase 4: Final high-precision polish - try all residues for all moduli
    if time.time() < deadline - 5:
        # Multiple passes of exact local search
        for pass_num in range(3):
            if time.time() > deadline - 5:
                break
            for i in range(len(classes)):
                if time.time() > deadline - 5:
                    break
                
                m, r = classes[i]
                
                # Try all residues exactly
                covered_temp = np.zeros(base, dtype=np.uint8)
                for j, (m2, r2) in enumerate(classes):
                    if j != i:
                        positions = np.arange(r2, base, m2)
                        covered_temp[positions] = 1
                
                base_uncovered = np.sum(covered_temp == 0)
                best_r2 = r
                best_uncovered = base_uncovered
                
                for r2 in range(m):
                    positions = np.arange(r2, base, m)
                    new_uncovered = base_uncovered - np.sum((covered_temp[positions] == 0).astype(np.uint8))
                    if new_uncovered < best_uncovered:
                        best_uncovered = new_uncovered
                        best_r2 = r2
                
                if best_r2 != r:
                    classes[i] = [m, best_r2]
    
    # Phase 5: Final removal pass
    if time.time() < deadline - 2:
        removed = True
        while removed and time.time() < deadline - 2:
            removed = False
            covered_current = np.zeros(base, dtype=np.uint8)
            for m, r in classes:
                positions = np.arange(r, base, m)
                covered_current[positions] = 1
            current_uncovered = np.sum(covered_current == 0)
            
            for i in range(len(classes) - 1, -1, -1):
                if time.time() > deadline - 2:
                    break
                covered_without = np.zeros(base, dtype=np.uint8)
                for j, (m, r) in enumerate(classes):
                    if j != i:
                        positions = np.arange(r, base, m)
                        covered_without[positions] = 1
                unc_without = np.sum(covered_without == 0)
                if unc_without <= current_uncovered:
                    classes.pop(i)
                    removed = True
                    current_uncovered = unc_without
    
    return classes
# EVOLVE-BLOCK-END
