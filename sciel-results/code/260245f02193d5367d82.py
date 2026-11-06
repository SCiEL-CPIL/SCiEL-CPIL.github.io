import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    def is_independent(mask, t_masks):
        for mt in t_masks:
            if (mt & mask) == mt:
                return False
        return True

    def compute_alpha(triples):
        t_masks = []
        for t in triples:
            m = 0
            for p in t:
                m |= (1 << p)
            t_masks.append(m)
        
        def find_max_indep(points_mask):
            if is_independent(points_mask, t_masks):
                return bin(points_mask).count('1')
            
            triple_found = None
            for mt in t_masks:
                if (mt & points_mask) == mt:
                    triple_found = mt
                    break
            
            if triple_found is None:
                return bin(points_mask).count('1')
            
            max_size = 0
            pts_in_triple = []
            for i in range(16):
                if triple_found & (1 << i):
                    pts_in_triple.append(i)
            
            for p in pts_in_triple:
                new_mask = points_mask & ~(1 << p)
                if bin(new_mask).count('1') <= max_size:
                    continue
                size = find_max_indep(new_mask)
                if size > max_size:
                    max_size = size
            
            return max_size

        return find_max_indep((1 << 16) - 1)

    def find_indep_set(points_mask, t_masks):
        if is_independent(points_mask, t_masks):
            return points_mask
        
        triple_found = None
        for mt in t_masks:
            if (mt & points_mask) == mt:
                triple_found = mt
                break
        
        if triple_found is None:
            return points_mask
        
        pts_in_triple = []
        for i in range(16):
            if triple_found & (1 << i):
                pts_in_triple.append(i)
        
        random.shuffle(pts_in_triple)
        
        best_set = 0
        for p in pts_in_triple:
            new_mask = points_mask & ~(1 << p)
            s = find_indep_set(new_mask, t_masks)
            if s and bin(s).count('1') > bin(best_set).count('1'):
                best_set = s
        return best_set

    # Construct STS(15)
    vectors = [
        0b0001, 0b0010, 0b0011, 0b0100, 0b0101, 
        0b0110, 0b0111, 0b1000, 0b1001, 0b1010, 
        0b1011, 0b1100, 0b1101, 0b1110, 0b1111
    ]
    vec_to_idx = {v: i for i, v in enumerate(vectors)}
    
    sts15_triples = []
    for i in range(15):
        for j in range(i+1, 15):
            v_k = vectors[i] ^ vectors[j]
            k = vec_to_idx[v_k]
            t = tuple(sorted([i, j, k]))
            sts15_triples.append(t)
            
    def can_add(triple, existing_triples):
        t_set = set(triple)
        for et in existing_triples:
            inter = t_set.intersection(et)
            if len(inter) > 1:
                return False
        return True

    best_alpha = 16
    best_triples = []
    
    for _ in range(500):
        if time.time() > deadline:
            break
            
        points_15 = list(range(15))
        random.shuffle(points_15)
        matching = []
        for i in range(0, 14, 2):
            matching.append((points_15[i], points_15[i+1]))
            
        initial_triples = []
        for t in sts15_triples:
            t_set = set(t)
            conflict = False
            for a, b in matching:
                if a in t_set and b in t_set:
                    conflict = True
                    break
            if not conflict:
                initial_triples.append(t)
                
        for a, b in matching:
            initial_triples.append((15, a, b))
            
        current_triples = initial_triples
        current_alpha = compute_alpha(current_triples)
        
        if current_alpha < best_alpha:
            best_alpha = current_alpha
            best_triples = current_triples[:]
            
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            alpha = compute_alpha(current_triples)
            if alpha < best_alpha:
                best_alpha = alpha
                best_triples = current_triples[:]
                
            t_masks = []
            for t in current_triples:
                m = 0
                for p in t:
                    m |= (1 << p)
                t_masks.append(m)
            
            indep_mask = find_indep_set((1<<16)-1, t_masks)
            indep_set = [i for i in range(16) if indep_mask & (1<<i)]
            
            added = False
            # Shuffle combinations to avoid bias?
            combos = list(itertools.combinations(indep_set, 3))
            random.shuffle(combos)
            
            for a, b, c in combos:
                t_new = (a,b,c)
                if can_add(t_new, current_triples):
                    current_triples.append(t_new)
                    added = True
                    improved = True
                    break
            
            if not added:
                break

    return [list(t) for t in best_triples]
