"""Combined program: chain-based initialization + structured local search with simulated annealing."""
import time
import random

_start_time = time.time()

def construct():
    deadline = _start_time + 85
    n = 12
    m = 40
    
    # Precompute subset sizes for all 4096 masks
    subset_sizes = [bin(i).count('1') for i in range(4096)]
    
    def make_chain_family():
        """Create initial family using multiple chains to maximize comparabilities."""
        sets = set()
        
        # Chain 1: add elements 0,1,2,...,11
        s = 0
        sets.add(s)
        for i in range(12):
            s |= (1 << i)
            sets.add(s)
        
        # Chain 2: add elements 1,2,...,11,0
        s = 0
        for v in [1,2,3,4,5,6,7,8,9,10,11,0]:
            s |= (1 << v)
            sets.add(s)
        
        # Chain 3: add elements 2,3,...,11,0,1
        s = 0
        for v in [2,3,4,5,6,7,8,9,10,11,0,1]:
            s |= (1 << v)
            sets.add(s)
        
        family = list(sets)
        seen = set(family)
        
        # Greedily add remaining sets to maximize comparabilities
        def score_family(fam):
            cnt = 0
            for i in range(len(fam)):
                for j in range(i+1, len(fam)):
                    a, b = fam[i], fam[j]
                    if a | b == a or a | b == b:
                        cnt += 1
            return cnt
        
        cur_score = score_family(family)
        
        all_masks = list(range(4096))
        random.shuffle(all_masks)
        
        while len(family) < 40:
            best_add = None
            best_deg = -1
            for cand in all_masks:
                if cand in seen:
                    continue
                deg = 0
                for s in family:
                    if cand | s == cand or cand | s == s:
                        deg += 1
                if deg > best_deg:
                    best_deg = deg
                    best_add = cand
            if best_add is not None:
                family.append(best_add)
                seen.add(best_add)
                cur_score += best_deg
            else:
                # Pick any unused mask
                for cand in all_masks:
                    if cand not in seen:
                        family.append(cand)
                        seen.add(cand)
                        break
                if len(family) >= 40:
                    break
        
        return family[:40]
    
    family = make_chain_family()
    
    def get_degree(mask, fam, exclude_idx=None):
        """Count how many sets in fam are comparable with mask."""
        cnt = 0
        for i, s in enumerate(fam):
            if i == exclude_idx:
                continue
            if mask | s == mask or mask | s == s:
                cnt += 1
        return cnt
    
    def compute_score(fam):
        cnt = 0
        for i in range(len(fam)):
            for j in range(i+1, len(fam)):
                a, b = fam[i], fam[j]
                if a | b == a or a | b == b:
                    cnt += 1
        return cnt
    
    cur_score = compute_score(family)
    best_family = list(family)
    best_score = cur_score
    
    random.seed(42)
    
    # Phase 1: Greedy local search - try to improve by swapping
    improved = True
    while improved:
        if time.time() > deadline - 10:
            break
        improved = False
        for i in range(40):
            if time.time() > deadline - 10:
                break
            old = family[i]
            deg_old = get_degree(old, family, i)
            
            # Try random candidates
            for _ in range(100):
                if time.time() > deadline - 10:
                    break
                new = random.randint(0, 4095)
                if new == old or new in family:
                    continue
                deg_new = 0
                for j, s in enumerate(family):
                    if j == i:
                        continue
                    if new | s == new or new | s == s:
                        deg_new += 1
                delta = deg_new - deg_old
                if delta > 0:
                    family[i] = new
                    cur_score += delta
                    improved = True
                    if cur_score > best_score:
                        best_score = cur_score
                        best_family = list(family)
                    break
    
    # Phase 2: Simulated annealing for escaping local optima
    temp = 5.0
    cooling = 0.95
    while temp > 0.1 and time.time() < deadline - 5:
        if time.time() > deadline - 5:
            break
        i = random.randint(0, 39)
        old = family[i]
        deg_old = get_degree(old, family, i)
        
        new = random.randint(0, 4095)
        if new == old or new in family:
            temp *= cooling
            continue
        
        deg_new = 0
        for j, s in enumerate(family):
            if j == i:
                continue
            if new | s == new or new | s == s:
                deg_new += 1
        
        delta = deg_new - deg_old
        if delta > 0 or random.random() < pow(2.718281828, delta / temp):
            family[i] = new
            cur_score += delta
            if cur_score > best_score:
                best_score = cur_score
                best_family = list(family)
        
        temp *= cooling
    
    # Phase 3: Final greedy polish
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        for i in range(40):
            if time.time() > deadline - 2:
                break
            old = family[i]
            deg_old = get_degree(old, family, i)
            
            for new in random.sample(range(4096), min(200, 4096)):
                if new == old or new in family:
                    continue
                deg_new = 0
                for j, s in enumerate(family):
                    if j == i:
                        continue
                    if new | s == new or new | s == s:
                        deg_new += 1
                delta = deg_new - deg_old
                if delta > 0:
                    family[i] = new
                    cur_score += delta
                    improved = True
                    if cur_score > best_score:
                        best_score = cur_score
                        best_family = list(family)
                    break
    
    # Convert to lists of ints
    result = []
    for mask in best_family:
        lst = [int(i) for i in range(12) if (mask >> i) & 1]
        result.append(lst)
    
    return result

if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    t1 = time.time()
    assert len(res) == 40
    seen = set()
    for r in res:
        fr = frozenset(r)
        assert fr not in seen
        seen.add(fr)
        for v in r:
            assert isinstance(v, int)
            assert 0 <= v <= 11
    masks = []
    for r in res:
        m = 0
        for v in r:
            m |= (1 << v)
        masks.append(m)
    cnt = 0
    for i in range(40):
        for j in range(i+1, 40):
            a, b = masks[i], masks[j]
            if a | b == a or a | b == b:
                cnt += 1
    print(f"Time: {t1-t0:.2f}s, Comparables: {cnt}")
