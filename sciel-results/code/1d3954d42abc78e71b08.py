import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    n = 512
    
    # Precompute all 4-APs
    # ap_list: list of tuples (a, b, c, d) where 1 <= a < b < c < d <= n
    # elem_to_aps: mapping from element to list of indices in ap_list
    ap_list = []
    elem_to_aps = [[] for _ in range(n + 1)]
    
    for a in range(1, n + 1):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            idx = len(ap_list)
            ap_list.append((a, b, c, e))
            elem_to_aps[a].append(idx)
            elem_to_aps[b].append(idx)
            elem_to_aps[c].append(idx)
            elem_to_aps[e].append(idx)
            
    num_aps = len(ap_list)

    def run_trial(seed):
        random.seed(seed)
        
        # 1. Greedy Hitting Set (Removal)
        # We want to remove elements to break all APs.
        # This is equivalent to finding a vertex cover in the hypergraph.
        # We use a greedy approach: repeatedly pick the element that covers the most active APs.
        
        in_set = [True] * (n + 1)
        active_aps = set(range(num_aps))
        
        # Calculate initial degrees (number of APs each element is in)
        deg = [0] * (n + 1)
        for i in range(num_aps):
            a, b, c, d = ap_list[i]
            deg[a] += 1
            deg[b] += 1
            deg[c] += 1
            deg[d] += 1
            
        while active_aps:
            # Find element with max degree among those in_set
            max_deg = -1
            candidates = []
            # Optimization: we can iterate only over elements that are still in set and have deg > 0
            # But simple loop is fine for n=512
            for i in range(1, n + 1):
                if in_set[i]:
                    if deg[i] > max_deg:
                        max_deg = deg[i]
                        candidates = [i]
                    elif deg[i] == max_deg:
                        candidates.append(i)
                        
            if not candidates or max_deg == 0:
                break
                
            # Random choice among max-degree candidates to add diversity
            x = random.choice(candidates)
            in_set[x] = False
            
            # Update degrees for all APs containing x
            for ap_idx in elem_to_aps[x]:
                if ap_idx in active_aps:
                    active_aps.remove(ap_idx)
                    a, b, c, d = ap_list[ap_idx]
                    for y in (a, b, c, d):
                        if y != x and in_set[y]:
                            deg[y] -= 1
                            
        S = set(i for i in range(1, n + 1) if in_set[i])
        
        # 2. Add-Back Phase
        # Try to add back removed elements if they don't create new APs.
        # This is a local search improvement.
        removed = [i for i in range(1, n + 1) if not in_set[i]]
        random.shuffle(removed)
        
        for x in removed:
            can_add = True
            for ap_idx in elem_to_aps[x]:
                a, b, c, d = ap_list[ap_idx]
                # Check if the other 3 elements are in S
                others = [y for y in (a, b, c, d) if y != x]
                if all(y in S for y in others):
                    can_add = False
                    break
            if can_add:
                S.add(x)
                
        # 3. Local Search (Swaps)
        # Try to improve S by swapping elements: remove one, add another (or two, etc.)
        # We will try simple swaps: remove x in S, add y not in S, if size increases or stays same but we allow it for exploration?
        # Actually, we want to maximize size. So we only accept if size increases.
        # But size can only increase by adding. Swaps keep size same.
        # To escape local optima, we might need to accept neutral or slightly worse moves?
        # Given time constraints, let's stick to strictly improving moves (additions) and maybe some random restarts.
        # The "Add-Back" phase already does a lot of work.
        
        # Let's do a quick pass of trying to add any remaining removed elements again?
        # No, the shuffle ensures we try them all once.
        
        return S

    best_set = set()
    best_size = 0
    
    # Run multiple trials
    # How many trials can we fit in 85 seconds?
    # Each trial involves O(N * A) operations roughly, where A is number of APs.
    # N=512, A ~ 10000.
    # The greedy removal loop runs until no active APs.
    # Let's estimate time per trial.
    
    trial_count = 0
    while time.time() < deadline:
        s = run_trial(trial_count)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
        trial_count += 1
        
    return sorted(best_set)
