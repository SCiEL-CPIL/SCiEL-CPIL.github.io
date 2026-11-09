import time
import random

def construct():
    deadline = time.time() + 140
    best_size = 0
    best_S = set()
    
    def evaluate(S, counts):
        # Verify validity and size
        # Not needed if logic is correct, but safe to return size
        return len(S)

    while time.time() < deadline - 5:
        # Greedy Initialization
        S = set()
        counts = {}
        
        def can_add(x):
            if counts.get(2*x, 0) >= 2: return False
            for y in S:
                if counts.get(x+y, 0) >= 2: return False
            return True
            
        def do_add(x):
            counts[2*x] = counts.get(2*x, 0) + 1
            for y in S:
                counts[x+y] = counts.get(x+y, 0) + 1
            S.add(x)
            
        def do_remove(x):
            S.remove(x)
            counts[2*x] -= 1
            if counts[2*x] == 0: del counts[2*x]
            for y in S:
                counts[x+y] -= 1
                if counts[x+y] == 0: del counts[x+y]
                
        # 1. Greedy Build - try multiple orderings
        for ordering in ['ascending', 'descending', 'random']:
            if ordering == 'ascending':
                candidates = list(range(1, 1001))
            elif ordering == 'descending':
                candidates = list(range(1000, 0, -1))
            else:
                candidates = list(range(1, 1001))
                random.shuffle(candidates)
                
            S.clear()
            counts.clear()
            for x in candidates:
                if can_add(x):
                    do_add(x)
                    
            if len(S) > best_size:
                best_size = len(S)
                best_S = set(S)
                
        # 2. Local Search with Kicks
        # Try to maximize size further
        while time.time() < deadline - 2:
            # First, try to fill remaining space with smart ordering
            added = False
            candidates = [x for x in range(1, 1001) if x not in S]
            
            # Sort by "slack" - elements that have more room to fit
            def get_slack(x):
                slack = 0
                if counts.get(2*x, 0) < 2:
                    slack += 2 - counts.get(2*x, 0)
                for y in S:
                    if counts.get(x+y, 0) < 2:
                        slack += 2 - counts.get(x+y, 0)
                return slack
            
            candidates.sort(key=get_slack, reverse=True)
            for x in candidates:
                if can_add(x):
                    do_add(x)
                    added = True
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = set(S)
                        
            if not added:
                # Stuck. Try a kick.
                # Randomly remove k elements
                if not S: break
                
                # Choose kick size. Small kicks are safer, large kicks can jump minima.
                k = random.randint(1, 8)
                if len(S) <= k:
                    k = max(1, len(S)//2)
                    
                to_remove = random.sample(list(S), k)
                
                # Save state
                S_copy = set(S)
                counts_copy = dict(counts)
                size_before = len(S)
                
                for x in to_remove:
                    do_remove(x)
                    
                # Now try to refill with smart ordering
                candidates = [x for x in range(1, 1001) if x not in S]
                candidates.sort(key=get_slack, reverse=True)
                for x in candidates:
                    if can_add(x):
                        do_add(x)
                        
                if len(S) > size_before:
                    # Accept new state (it's better)
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = set(S)
                    # Continue loop to try more adds
                else:
                    # Revert
                    S.clear()
                    S.update(S_copy)
                    counts.clear()
                    counts.update(counts_copy)
                    
                    # If we keep reverting, we might be in a very tight local optimum.
                    # To escape, we might need a larger kick or accept a worse state temporarily.
                    # For now, let's try another random kick.
                    # If we get stuck in reverts for a long time, the deadline will kill us.
                    
    # Final polish: try to add any remaining elements
    final_S = set(best_S)
    final_counts = {}
    for x in final_S:
        final_counts[2*x] = final_counts.get(2*x, 0) + 1
        for y in final_S:
            if x != y:
                final_counts[x+y] = final_counts.get(x+y, 0) + 1
    
    for x in range(1, 1001):
        if x not in final_S:
            if final_counts.get(2*x, 0) < 2:
                ok = True
                for y in final_S:
                    if final_counts.get(x+y, 0) >= 2:
                        ok = False
                        break
                if ok:
                    final_S.add(x)
                    final_counts[2*x] = final_counts.get(2*x, 0) + 1
                    for y in final_S:
                        if y != x:
                            final_counts[x+y] = final_counts.get(x+y, 0) + 1
    
    if len(final_S) > best_size:
        best_size = len(final_S)
        best_S = final_S
        
    return sorted(best_S)
