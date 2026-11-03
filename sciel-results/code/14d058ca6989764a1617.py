import time
import sys

def construct(n):
    if n == 1:
        return "1"
    
    word = [1]
    
    for m in range(2, n + 1):
        # 1. Extract unique permutations of length m-1 from word
        perms = []
        seen = set()
        # The windows of length m-1 in the previous word
        # Note: The previous word was constructed for m-1 symbols.
        # We need to find all unique (m-1)-permutations.
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # 2. Generate blocks for each perm
        blocks = []
        for p in perms:
            b = list(p) + [m] + list(p)
            blocks.append(b)
            
        # 3. Compute overlap matrix
        num_perms = len(perms)
        # overlap[i][j] = max k such that blocks[i][-k:] == blocks[j][:k]
        # We want to minimize length, so cost = len(block) - overlap.
        # We can just store overlap and use it in TSP.
        
        # Optimization: Precompute string representations for faster overlap check
        block_strs = ["".join(map(str, b)) for b in blocks]
        max_len = len(blocks[0])
        
        overlaps = [[0] * num_perms for _ in range(num_perms)]
        
        for i in range(num_perms):
            bi = block_strs[i]
            for j in range(num_perms):
                if i == j:
                    overlaps[i][j] = 0
                    continue
                bj = block_strs[j]
                # Max possible overlap is min(len(bi), len(bj))
                # But since bi ends with p_i and bj starts with p_j,
                # and p_i, p_j are permutations, overlap is likely small.
                # Check k from min(max_len, len(bi), len(bj)) down to 0
                # Actually, we can just check up to m-1?
                # The suffix of bi is ... p_i. Prefix of bj is p_j ...
                # If p_i and p_j share a suffix/prefix, we can overlap.
                max_k = min(max_len, len(bi), len(bj))
                # Optimization: only check k up to m (since the repeating part is m-1)
                # The structure is p m p.
                # Suffix of bi: ... p_i[m-1-k+1:] + ...?
                # Let's just check k from max_k down to 0.
                for k in range(max_k, -1, -1):
                    if bi[-k:] == bj[:k]:
                        overlaps[i][j] = k
                        break
                        
        # 4. Solve TSP to order the perms
        # Cost matrix: cost[i][j] = len(blocks[i]) - overlaps[i][j]
        # We start at 0, visit all, no return to 0 (Path, not Cycle)
        
        # Heuristic: Nearest Neighbor
        unvisited = set(range(num_perms))
        path = [0]
        unvisited.remove(0)
        current = 0
        
        while unvisited:
            best_next = -1
            best_cost = float('inf')
            for next_node in unvisited:
                cost = len(blocks[current]) - overlaps[current][next_node]
                # Note: The first block adds len(blocks[0]) - overlap_with_start?
                # The seed program handles the initial append differently.
                # In the seed program, `new` starts empty.
                # First block: overlap with empty is 0.
                # Subsequent blocks: overlap with previous.
                # So the cost of the path is sum(len(block_i)) - sum(overlap(prev, curr)).
                # This is equivalent to minimizing sum(cost[i][j]).
                
                if cost < best_cost:
                    best_cost = cost
                    best_next = next_node
            
            path.append(best_next)
            unvisited.remove(best_next)
            current = best_next
            
        # 5. Construct new word
        new_word = []
        for idx, p_idx in enumerate(path):
            b = blocks[p_idx]
            if idx == 0:
                new_word.extend(b)
            else:
                prev_b = blocks[path[idx-1]]
                # Find overlap
                # We already have overlaps[prev_idx][curr_idx]
                ov = overlaps[path[idx-1]][p_idx]
                # Append b[ov:]
                new_word.extend(b[ov:])
                
        word = new_word
        
    return "".join(map(str, word))
