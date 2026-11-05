"""Covering design C(24,7,3) - efficient greedy + local search."""
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110
    
    all_points = list(range(v))
    all_t_subsets = list(itertools.combinations(all_points, t))
    num_t = len(all_t_subsets)
    t_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Greedy construction
    uncovered = set(range(num_t))
    blocks = []
    
    def get_covered(block):
        return [t_to_idx[ts] for ts in itertools.combinations(block, t)]
    
    while uncovered and time.time() < deadline:
        # Pick a random uncovered t-subset as seed
        sample = random.choice(list(uncovered)[:min(len(uncovered), 50)])
        seed = list(all_t_subsets[sample])
        
        block = list(seed)
        remaining = [p for p in all_points if p not in block]
        
        while len(block) < k:
            best_x, best_gain = None, -1
            for x in remaining:
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    ts = tuple(sorted(rest + (x,)))
                    idx = t_to_idx.get(ts, -1)
                    if idx >= 0 and idx in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain, best_x = gain, x
            if best_x is None or best_gain == 0:
                # No improvement possible, just fill randomly
                block.append(random.choice(remaining))
                remaining.remove(block[-1])
            else:
                block.append(best_x)
                remaining.remove(best_x)
        
        block.sort()
        covered = get_covered(block)
        new_cov = [i for i in covered if i in uncovered]
        if new_cov:
            blocks.append(block)
            for i in new_cov:
                uncovered.discard(i)
    
    # Local search: try to reduce block count
    for _ in range(30):
        if time.time() > deadline:
            break
        improved = False
        
        # Try removing each block
        for i in range(len(blocks)):
            if time.time() > deadline:
                break
            test = blocks[:i] + blocks[i+1:]
            cov = set()
            for b in test:
                cov.update(get_covered(b))
            if len(cov) == num_t:
                blocks = test
                improved = True
                break
        
        if improved:
            continue
        
        # Try replacing a block
        for i in range(len(blocks)):
            if time.time() > deadline:
                break
            without = blocks[:i] + blocks[i+1:]
            cov_without = set()
            for b in without:
                cov_without.update(get_covered(b))
            must = set(range(num_t)) - cov_without
            
            if not must:
                blocks = without
                improved = True
                break
            
            # Find required points
            req = set()
            for idx in must:
                req.update(all_t_subsets[idx])
            
            if len(req) > k:
                continue
            
            slots = k - len(req)
            cands = [p for p in all_points if p not in req]
            
            for _ in range(15):
                if time.time() > deadline:
                    break
                if slots <= 0:
                    nb = tuple(sorted(req))
                else:
                    extra = random.sample(cands, min(slots, len(cands)))
                    nb = tuple(sorted(req | set(extra)))
                
                ncov = set(get_covered(list(nb)))
                if must.issubset(ncov) and nb not in [tuple(sorted(b)) for b in without]:
                    blocks = without + [list(nb)]
                    improved = True
                    break
            
            if improved:
                break
        
        if not improved:
            break
    
    return blocks
