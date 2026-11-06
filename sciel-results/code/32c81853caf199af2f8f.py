import time
import random

def construct():
    d = 7
    n = 1 << d
    deadline = time.time() + 130
    
    # Precompute all edges
    all_edges = []
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                all_edges.append((x, y))
    
    n_edges = len(all_edges)
    edge_idx = {e: i for i, e in enumerate(all_edges)}
    
    # Precompute faces
    faces = []
    edge_to_face = [[] for _ in range(n_edges)]
    
    for i in range(d):
        for j in range(i + 1, d):
            for mask in range(1 << (d - 2)):
                v0 = 0
                bp = 0
                for k in range(d):
                    if k != i and k != j:
                        if (mask >> bp) & 1:
                            v0 |= (1 << k)
                        bp += 1
                
                v1 = v0 ^ (1 << i)
                v2 = v0 ^ (1 << j)
                v3 = v0 ^ (1 << i) ^ (1 << j)
                
                e1 = (min(v0, v1), max(v0, v1))
                e2 = (min(v0, v2), max(v0, v2))
                e3 = (min(v1, v3), max(v1, v3))
                e4 = (min(v2, v3), max(v2, v3))
                
                idxs = [edge_idx[e1], edge_idx[e2], edge_idx[e3], edge_idx[e4]]
                fidx = len(faces)
                faces.append(idxs)
                for idx in idxs:
                    edge_to_face[idx].append(fidx)
    
    n_faces = len(faces)
    
    def greedy_fill(selected, rng):
        """Add edges greedily in random order."""
        changed = True
        while changed:
            changed = False
            order = list(range(n_edges))
            rng.shuffle(order)
            for idx in order:
                if selected[idx]:
                    continue
                ok = True
                for fidx in edge_to_face[idx]:
                    face = faces[fidx]
                    if all(selected[i] for i in face if i != idx):
                        ok = False
                        break
                if ok:
                    selected[idx] = True
                    changed = True
    
    def local_search(selected, rng):
        """Local search with swaps."""
        count = sum(selected)
        no_improve = 0
        max_no_improve = 50
        
        while no_improve < max_no_improve:
            if time.time() > deadline:
                break
            
            # Try to add an edge
            improved = False
            order = list(range(n_edges))
            rng.shuffle(order)
            for idx in order:
                if selected[idx]:
                    continue
                ok = True
                for fidx in edge_to_face[idx]:
                    face = faces[fidx]
                    if all(selected[i] for i in face if i != idx):
                        ok = False
                        break
                if ok:
                    selected[idx] = True
                    count += 1
                    improved = True
                    break
            
            if improved:
                no_improve = 0
                continue
            
            no_improve += 1
            
            # Try a swap: remove one edge, try to add others
            removable = [i for i in range(n_edges) if selected[i]]
            if not removable:
                break
            
            # Pick an edge that's in many faces (high degree)
            # Or just random
            rem = rng.choice(removable)
            selected[rem] = False
            count -= 1
            
            # Try to add edges
            added = 0
            order2 = list(range(n_edges))
            rng.shuffle(order2)
            added_edges = []
            for idx in order2:
                if selected[idx]:
                    continue
                ok = True
                for fidx in edge_to_face[idx]:
                    face = faces[fidx]
                    if all(selected[i] for i in face if i != idx):
                        ok = False
                        break
                if ok:
                    selected[idx] = True
                    added += 1
                    added_edges.append(idx)
            
            if added <= 1:
                # Revert
                selected[rem] = True
                count += 1
                for idx in added_edges:
                    selected[idx] = False
                    count -= 1
            else:
                no_improve = 0
        
        return count
    
    def run_one_trial(rng):
        """Run one trial."""
        selected = [False] * n_edges
        
        # Random initialization
        density = rng.uniform(0.3, 0.6)
        for idx in range(n_edges):
            if rng.random() < density:
                selected[idx] = True
        
        # Fix violations
        for _ in range(200):
            found = False
            for face in faces:
                if all(selected[idx] for idx in face):
                    rem = rng.choice(face)
                    selected[rem] = False
                    found = True
                    break
            if not found:
                break
        
        # Greedy fill
        greedy_fill(selected, rng)
        
        # Local search
        count = local_search(selected, rng)
        
        return selected, count
    
    best_selected = None
    best_count = 0
    rng = random.Random(12345)
    
    trials = 0
    while time.time() < deadline:
        trials += 1
        selected, count = run_one_trial(rng)
        if count > best_count:
            best_count = count
            best_selected = selected[:]
        # Change seed periodically
        if trials % 20 == 0:
            rng = random.Random(rng.randint(0, 10**9))
    
    if best_selected is None:
        # Fallback
        selected = [False] * n_edges
        greedy_fill(selected, random.Random(0))
        best_selected = selected
        best_count = sum(selected)
    
    out = [list(all_edges[idx]) for idx in range(n_edges) if best_selected[idx]]
    return out
