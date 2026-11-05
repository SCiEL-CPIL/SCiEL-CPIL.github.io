import time
import random
import itertools

# EVOLVE-BLOCK-START
def construct(n):
    if n <= 1:
        return "1"
    
    # Generate the standard recursive word to get the list of blocks for n-1
    # We need the list of distinct (n-1)-permutations in the order of the standard construction?
    # No, the set of blocks is just all permutations of 1..n-1.
    # The recursive construction uses the blocks in the order they appear.
    # But for the TSP approach, we just need the set of all permutations of 1..n-1.
    
    # Let's generate all permutations of 1..n-1
    symbols = list(range(1, n))
    perms = list(itertools.permutations(symbols))
    K = len(perms)
    
    # If n is small, we can even try to solve TSP exactly or use a very strong heuristic
    # For n=7, K=720.
    
    # Precompute overlap matrix
    # overlap[i][j] = length of longest suffix of perms[i] that is a prefix of perms[j]
    # Max possible overlap is n-2 (since length is n-1)
    # Actually, if perms[i] == perms[j], overlap is n-1. But we visit each once, so i!=j.
    # Max overlap for distinct perms is n-2.
    
    overlap = [[0] * K for _ in range(K)]
    for i in range(K):
        u = perms[i]
        for j in range(K):
            if i == j: continue
            v = perms[j]
            max_ov = n - 2 # max possible for distinct
            # Check suffix of u of length L vs prefix of v of length L
            # L from max_ov down to 0
            # Optimization: only check L where u[-L:] == v[:L]
            # Since n is small (<=7), simple loop is fine
            found = 0
            # Max overlap is n-2. 
            # e.g. n=7, len=6. max ov=5.
            for L in range(n - 2, -1, -1):
                if u[-L:] == v[:L]:
                    found = L
                    break
            overlap[i][j] = found
            
    # Heuristic to find a good path
    # Greedy + Local Search
    
    def greedy_path(start_idx):
        unvisited = set(range(K))
        unvisited.remove(start_idx)
        path = [start_idx]
        curr = start_idx
        while unvisited:
            best_next = -1
            best_ov = -1
            for v in unvisited:
                if overlap[curr][v] > best_ov:
                    best_ov = overlap[curr][v]
                    best_next = v
            path.append(best_next)
            unvisited.remove(best_next)
            curr = best_next
        return path

    def path_score(path):
        score = 0
        for i in range(len(path) - 1):
            score += overlap[path[i]][path[i+1]]
        return score

    def two_opt(path):
        improved = True
        while improved:
            improved = False
            for i in range(len(path) - 2):
                for j in range(i + 1, len(path)):
                    # Reverse path[i+1 ... j]
                    # Old edges: (path[i-1], path[i]) and (path[j], path[j+1])
                    # Note: if i==0, no left edge. If j==K-1, no right edge.
                    # But we are constructing a path, not a cycle.
                    # Let's be careful with boundaries.
                    
                    # Actually, standard 2-opt on path:
                    # Segment [i+1, j] is reversed.
                    # Connections change at i -> i+1 and j -> j+1.
                    
                    # Let a = path[i], b = path[i+1], c = path[j], d = path[j+1]
                    # If i==0, a is start.
                    # If j==K-1, d is end.
                    
                    # New connections: a -> c and b -> d
                    # Old connections: a -> b and c -> d
                    
                    ov_old = 0
                    if i > 0:
                        ov_old += overlap[path[i-1]][path[i]] # Wait, this is not the edge being cut
                        # The edge being cut is (path[i-1], path[i])? No.
                        # The path is ... path[i-1], path[i], path[i+1] ... path[j], path[j+1] ...
                        # We reverse path[i+1] ... path[j].
                        # The new sequence is ... path[i-1], path[i], path[j], path[j-1] ... path[i+1], path[j+1] ...
                        # So the edges (path[i], path[i+1]) and (path[j], path[j+1]) are removed.
                        # New edges are (path[i], path[j]) and (path[i+1], path[j+1]).
                        
                        pass # Logic below
                    
                    if i == 0:
                        # No left edge to consider for cut?
                        # The start is path[0].
                        # If i=0, we reverse path[1...j].
                        # Old: path[0]->path[1], path[j]->path[j+1]
                        # New: path[0]->path[j], path[1]->path[j+1]
                        old_cost = overlap[path[0]][path[1]] + (overlap[path[j]][path[j+1]] if j < K-1 else 0)
                        new_cost = overlap[path[0]][path[j]] + (overlap[path[1]][path[j+1]] if j < K-1 else 0)
                    else:
                        old_cost = overlap[path[i]][path[i+1]] + (overlap[path[j]][path[j+1]] if j < K-1 else 0)
                        new_cost = overlap[path[i]][path[j]] + (overlap[path[i+1]][path[j+1]] if j < K-1 else 0)
                        
                    if new_cost > old_cost:
                        # Perform reversal
                        path[i+1:j+1] = reversed(path[i+1:j+1])
                        improved = True
        return path

    # Try multiple random starts
    best_path = None
    best_score = -1
    start_time = time.time()
    deadline = start_time + 100 # Leave 20s margin
    
    # First try the standard order?
    # The standard order is the order of appearance in the recursive word.
    # We can reconstruct that order or just use random.
    # Random is fine.
    
    for _ in range(100):
        if time.time() > deadline:
            break
        start_idx = random.randint(0, K - 1)
        p = greedy_path(start_idx)
        p = two_opt(p)
        s = path_score(p)
        if s > best_score:
            best_score = s
            best_path = p
            
    # If time remains, do more 2-opt iterations on the best path found
    if best_path is not None and time.time() < deadline:
        # Intensive local search
        while time.time() < deadline - 5:
            old_path = list(best_path)
            # Do one pass of 2-opt
            improved = False
            for i in range(len(best_path) - 2):
                if time.time() > deadline - 5: break
                for j in range(i + 1, len(best_path)):
                    if i == 0:
                        old_cost = overlap[best_path[0]][best_path[1]] + (overlap[best_path[j]][best_path[j+1]] if j < K-1 else 0)
                        new_cost = overlap[best_path[0]][best_path[j]] + (overlap[best_path[1]][best_path[j+1]] if j < K-1 else 0)
                    else:
                        old_cost = overlap[best_path[i]][best_path[i+1]] + (overlap[best_path[j]][best_path[j+1]] if j < K-1 else 0)
                        new_cost = overlap[best_path[i]][best_path[j]] + (overlap[best_path[i+1]][best_path[j+1]] if j < K-1 else 0)
                    
                    if new_cost > old_cost:
                        best_path[i+1:j+1] = reversed(best_path[i+1:j+1])
                        improved = True
            if not improved:
                break
        s = path_score(best_path)
        if s > best_score:
            best_score = s

    if best_path is None:
        # Fallback to standard construction
        return _standard_construct(n)

    # Construct the word
    word = list(perms[best_path[0]])
    for idx in best_path[1:]:
        p = perms[idx]
        # Append p, but overlap with previous
        # The previous block was perms[best_path[-2]]
        # We already accounted for overlap in the score?
        # No, we need to build the string.
        # The string is formed by concatenating blocks with 'n' inserted.
        # Block i: perms[best_path[i]]
        # Transition: overlap suffix of block i-1 with prefix of block i.
        # We append the non-overlapping part of block i, then 'n'.
        
        prev_block = word # This is the whole word so far? No.
        # It's easier to just build the full string by iterating.
        
    # Let's rebuild properly
    # Start with first block
    final_word = list(perms[best_path[0]])
    for k in range(1, K):
        curr_block = perms[best_path[k]]
        prev_block = perms[best_path[k-1]]
        # Find overlap
        # We stored overlap[best_path[k-1]][best_path[k]]
        ov = overlap[best_path[k-1]][best_path[k]]
        # Append curr_block[ov:]
        final_word.extend(curr_block[ov:])
        # Append n
        final_word.append(n)
        
    return "".join(map(str, final_word))

def _standard_construct(n):
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
