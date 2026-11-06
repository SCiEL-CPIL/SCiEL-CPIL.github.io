import time
import random
import itertools

def construct():
    n = 20
    random.seed(42)
    
    # Edge mapping
    edge_index = {}
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_index[(u, v)] = idx
            idx += 1
    total_edges = idx # 190
    
    # Precompute forbidden K3,3 masks
    triples = list(itertools.combinations(range(n), 3))
    forbidden_masks = []
    for i in range(len(triples)):
        t1 = triples[i]
        set_t1 = set(t1)
        for j in range(i + 1, len(triples)):
            t2 = triples[j]
            if set_t1.isdisjoint(set(t2)):
                mask = 0
                for u in t1:
                    for v in t2:
                        if u > v:
                            u, v = v, u
                        mask |= (1 << edge_index[(u, v)])
                forbidden_masks.append(mask)
                
    # Shuffle masks to avoid bias in finding violations
    random.shuffle(forbidden_masks)
    
    best_mask = 0
    best_count = 0
    deadline = time.time() + 110.0
    
    def popcount(x):
        return bin(x).count('1')
        
    def find_violation(mask):
        for fm in forbidden_masks:
            if (mask & fm) == fm:
                return fm
        return None
        
    def get_edges_from_mask(mask):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                e_idx = edge_index[(u, v)]
                if (mask >> e_idx) & 1:
                    edges.append([u, v])
        return edges

    # Phase 1: Heuristic to get a good lower bound
    # Start with full graph, remove edges to satisfy constraints
    def heuristic_search():
        nonlocal best_mask, best_count
        current_mask = (1 << total_edges) - 1
        # Greedy removal
        while True:
            if time.time() > deadline:
                return
            vm = find_violation(current_mask)
            if vm is None:
                count = popcount(current_mask)
                if count > best_count:
                    best_count = count
                    best_mask = current_mask
                return
            # Remove a random edge from violation
            edges_in_vm = []
            tmp = vm
            while tmp:
                lsb = tmp & -tmp
                idx = lsb.bit_length() - 1
                edges_in_vm.append(idx)
                tmp ^= lsb
            e_idx = random.choice(edges_in_vm)
            current_mask &= ~(1 << e_idx)
            
    # Run heuristic for a short time
    heuristic_deadline = time.time() + 5.0
    while time.time() < heuristic_deadline:
        heuristic_search()
        
    # Phase 2: Branch and Bound
    # Use best_count from Phase 1 as initial lower bound
    
    def search(current_mask, depth):
        nonlocal best_mask, best_count
        
        if time.time() > deadline:
            return
            
        current_count = popcount(current_mask)
        if current_count <= best_count:
            return
            
        vm = find_violation(current_mask)
        
        if vm is None:
            if current_count > best_count:
                best_count = current_count
                best_mask = current_mask
            return
            
        # Branch on edges in violation
        edges_in_vm = []
        tmp = vm
        while tmp:
            lsb = tmp & -tmp
            idx = lsb.bit_length() - 1
            edges_in_vm.append(idx)
            tmp ^= lsb
            
        # Sort edges by degree in current graph? Or just random order?
        # Random order helps escape local minima in search
        random.shuffle(edges_in_vm)
        
        for e_idx in edges_in_vm:
            new_mask = current_mask & ~(1 << e_idx)
            search(new_mask, depth + 1)
            if time.time() > deadline:
                return

    search((1 << total_edges) - 1, 0)
    
    return get_edges_from_mask(best_mask)

if __name__ == "__main__":
    pass
