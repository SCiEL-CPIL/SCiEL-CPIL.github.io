import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Precompute masks and vertices for size 6 subsets for n=22
    # We will handle n=21 and n=22 separately if needed, but let's start with 22
    
    def generate_masks(n, k):
        masks = []
        verts_list = []
        for subset in combinations(range(n), k):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            masks.append(mask)
            verts_list.append(subset)
        return masks, verts_list

    # Precompute for n=22, k=6
    masks_6_22, verts_6_22 = generate_masks(22, 6)
    
    def score_6(adj, masks, verts_list):
        count = 0
        for i, mask in enumerate(masks):
            verts = verts_list[i]
            degs = [(adj[v] & mask).bit_count() for v in verts]
            if len(set(degs)) == 1:
                count += 1
        return count

    def is_valid_full(adj, n):
        # Check all subsets of size >= 6
        # This is expensive, only call when score_6 is 0
        for k in range(6, n + 1):
            for subset in combinations(range(n), k):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                degs = [(adj[v] & mask).bit_count() for v in subset]
                if len(set(degs)) == 1:
                    return False
        return True

    # Search for n=22
    best_adj_22 = None
    best_score_22 = float('inf')
    
    for restart in range(10):
        if time.time() > deadline - 20:
            break
            
        adj = [0] * 22
        for i in range(22):
            for j in range(i + 1, 22):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    
        current_score = score_6(adj, masks_6_22, verts_6_22)
        if current_score < best_score_22:
            best_score_22 = current_score
            best_adj_22 = adj[:]
            
        if current_score == 0:
            if is_valid_full(adj, 22):
                edges = []
                for i in range(22):
                    for j in range(i + 1, 22):
                        if (adj[i] & (1 << j)) != 0:
                            edges.append([i, j])
                return {'n': 22, 'edges': edges}
            # If not valid, we might try to fix it, but for now restart
            continue

        # Local search
        no_improve = 0
        while current_score > 0 and no_improve < 200 and time.time() < deadline - 20:
            u = random.randint(0, 21)
            v = random.randint(0, 21)
            if u == v: continue
            
            # Flip
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            
            new_score = score_6(adj, masks_6_22, verts_6_22)
            
            if new_score < current_score:
                current_score = new_score
                no_improve = 0
                if current_score < best_score_22:
                    best_score_22 = current_score
                    best_adj_22 = adj[:]
                if current_score == 0:
                    break
            else:
                # Revert
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                no_improve += 1
                # Annealing
                if random.random() < 0.05:
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                    current_score = new_score
                    no_improve = 0

        # Check if we hit 0
        if current_score == 0:
            if is_valid_full(adj, 22):
                edges = []
                for i in range(22):
                    for j in range(i + 1, 22):
                        if (adj[i] & (1 << j)) != 0:
                            edges.append([i, j])
                return {'n': 22, 'edges': edges}

    # If 22 failed, try 21
    if time.time() < deadline - 10:
        masks_6_21, verts_6_21 = generate_masks(21, 6)
        
        for restart in range(10):
            if time.time() > deadline - 10:
                break
                
            adj = [0] * 21
            for i in range(21):
                for j in range(i + 1, 21):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                        
            current_score = score_6(adj, masks_6_21, verts_6_21)
            
            if current_score == 0:
                if is_valid_full(adj, 21):
                    edges = []
                    for i in range(21):
                        for j in range(i + 1, 21):
                            if (adj[i] & (1 << j)) != 0:
                                edges.append([i, j])
                    return {'n': 21, 'edges': edges}
                continue

            no_improve = 0
            while current_score > 0 and no_improve < 200 and time.time() < deadline - 10:
                u = random.randint(0, 20)
                v = random.randint(0, 20)
                if u == v: continue
                
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                
                new_score = score_6(adj, masks_6_21, verts_6_21)
                
                if new_score < current_score:
                    current_score = new_score
                    no_improve = 0
                    if current_score == 0:
                        break
                else:
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                    no_improve += 1
                    if random.random() < 0.05:
                        adj[u] ^= (1 << v)
                        adj[v] ^= (1 << u)
                        current_score = new_score
                        no_improve = 0

            if current_score == 0:
                if is_valid_full(adj, 21):
                    edges = []
                    for i in range(21):
                        for j in range(i + 1, 21):
                            if (adj[i] & (1 << j)) != 0:
                                edges.append([i, j])
                    return {'n': 21, 'edges': edges}

    # Fallback
    return {'n': 5, 'edges': []}
