import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_score = 0.0
    best_classes = []
    
    def score_of(classes):
        return sum(1.0 / m for m, r in classes)
    
    def is_valid(classes):
        n = len(classes)
        for i in range(n):
            for j in range(i + 1, n):
                m1, r1 = classes[i]
                m2, r2 = classes[j]
                g = gcd(m1, m2)
                if r1 % g == r2 % g:
                    return False
        return True
    
    def greedy_construct(order, residue_strategy="smallest"):
        """Build classes greedily given an order of moduli and residue strategy."""
        classes = []
        for m in order:
            if time.time() > deadline:
                break
            found = False
            if residue_strategy == "smallest":
                candidates = range(m)
            elif residue_strategy == "largest":
                candidates = range(m - 1, -1, -1)
            elif residue_strategy == "random":
                candidates = list(range(m))
                random.shuffle(candidates)
            else:
                candidates = range(m)
            
            for r in candidates:
                ok = True
                for m2, r2 in classes:
                    g = gcd(m, m2)
                    if r % g == r2 % g:
                        ok = False
                        break
                if ok:
                    classes.append([m, r])
                    found = True
                    break
        return classes
    
    def local_search(classes, max_iters=5000):
        """Try to improve by adjusting residues or adding skipped moduli."""
        classes = [list(c) for c in classes]
        selected_moduli = set(m for m, r in classes)
        all_moduli = set(range(20, 201))
        skipped = sorted(all_moduli - selected_moduli)
        
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            
            # Try to add a skipped modulus by adjusting existing residues
            for m_new in skipped[:20]:  # Only try small skipped moduli (high value)
                if time.time() > deadline:
                    break
                # Find a residue for m_new
                best_r = None
                for r in range(m_new):
                    ok = True
                    for idx, (m2, r2) in enumerate(classes):
                        g = gcd(m_new, m2)
                        if r % g == r2 % g:
                            ok = False
                            break
                    if ok:
                        best_r = r
                        break
                
                if best_r is not None:
                    classes.append([m_new, best_r])
                    selected_moduli.add(m_new)
                    skipped.remove(m_new)
                    improved = True
                    continue
                
                # Try adjusting one existing residue to free up space
                for idx in range(len(classes)):
                    m2, r2 = classes[idx]
                    # Try changing r2 to see if that frees up a residue for m_new
                    for r_new in range(m_new):
                        # Check if r_new conflicts with any class except idx
                        ok_new = True
                        for jdx, (mj, rj) in enumerate(classes):
                            if jdx == idx:
                                continue
                            g = gcd(m_new, mj)
                            if r_new % g == rj % g:
                                ok_new = False
                                break
                        if not ok_new:
                            continue
                        
                        # Check if we can change r2 to some r2_new that's compatible
                        # with all other classes (not m_new since we haven't added it yet)
                        for r2_new in range(m2):
                            ok_adj = True
                            for jdx, (mj, rj) in enumerate(classes):
                                if jdx == idx:
                                    continue
                                g = gcd(m2, mj)
                                if r2_new % g == rj % g:
                                    ok_adj = False
                                    break
                            if not ok_adj:
                                continue
                            
                            # Check r2_new doesn't conflict with m_new at r_new
                            g = gcd(m2, m_new)
                            if r2_new % g == r_new % g:
                                continue
                            
                            # Apply the change
                            old_r2 = classes[idx][1]
                            classes[idx][1] = r2_new
                            classes.append([m_new, r_new])
                            selected_moduli.add(m_new)
                            if m_new in skipped:
                                skipped.remove(m_new)
                            improved = True
                            break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    continue
            
            # Also try adjusting residues of existing classes to allow adding another
            # (this is covered above, but let's also try pure residue swaps)
        
        return classes
    
    # Strategy 1: Greedy with smallest moduli first
    order1 = list(range(20, 201))
    classes1 = greedy_construct(order1, "smallest")
    s1 = score_of(classes1)
    if s1 > best_score:
        best_score = s1
        best_classes = [list(c) for c in classes1]
    
    # Strategy 2: Greedy with largest moduli first (to see if it helps)
    order2 = list(range(200, 19, -1))
    classes2 = greedy_construct(order2, "smallest")
    s2 = score_of(classes2)
    if s2 > best_score:
        best_score = s2
        best_classes = [list(c) for c in classes2]
    
    # Strategy 3: Randomized restarts
    random.seed(42)
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Randomize the order slightly
        order = list(range(20, 201))
        random.shuffle(order)
        # But keep small moduli somewhat prioritized
        # Sort by a mix of size and random
        order.sort(key=lambda m: m + random.random() * 30)
        
        strategy = random.choice(["smallest", "largest", "random"])
        classes = greedy_construct(order, strategy)
        
        # Local search
        classes = local_search(classes, max_iters=2000)
        
        s = score_of(classes)
        if s > best_score:
            best_score = s
            best_classes = [list(c) for c in classes]
    
    # Final local search on best
    if time.time() < deadline:
        improved_classes = local_search(best_classes, max_iters=10000)
        s = score_of(improved_classes)
        if s > best_score:
            best_score = s
            best_classes = [list(c) for c in improved_classes]
    
    return best_classes
