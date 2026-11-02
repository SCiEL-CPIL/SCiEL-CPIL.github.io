"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import itertools
import random
import time
from collections import defaultdict, deque

def construct(q, n, radius):
    # New approach: Greedy initialization + Maximum Matching-based local search
    # Uses Hopcroft-Karp to find minimum vertex covers in bipartite graphs
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters
        all_words = list(itertools.product(range(q), repeat=n))
        uncovered = set(all_words)
        code = []
        while uncovered:
            best_word = None
            best_count = -1
            for w in all_words:
                count = sum(1 for u in uncovered if sum(a != b for a, b in zip(w, u)) <= radius)
                if count > best_count:
                    best_count = count
                    best_word = w
            code.append(list(best_word))
            uncovered = {u for u in uncovered if sum(a != b for a, b in zip(best_word, u)) > radius}
        return code

    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx

    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    total_words = q ** n
    indices = list(range(total_words))
    
    # Precompute neighborhoods
    neighborhoods = []
    for i in indices:
        w = idx_to_word(i)
        neigh = [i]
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = w.copy()
                new_w[pos] = val
                neigh.append(word_to_idx(new_w))
        neighborhoods.append(neigh)
    
    # Build reverse neighborhoods: for each word, which codewords cover it
    reverse_neigh = [[] for _ in range(total_words)]
    for i in range(total_words):
        for nb in neighborhoods[i]:
            reverse_neigh[nb].append(i)
    
    # Phase 1: Greedy Set Cover with lookahead
    uncovered = set(indices)
    code_indices = []
    
    while uncovered:
        # Score each candidate by how many uncovered words it covers
        # Use a weighted score to break ties
        best_idx = -1
        best_score = -1
        
        for i in indices:
            count = sum(1 for nb in neighborhoods[i] if nb in uncovered)
            if count == 0:
                continue
            # Tie-breaker: prefer codewords that cover words with few alternatives
            alt_count = sum(1 for nb in neighborhoods[i] if nb in uncovered)
            rareness = sum(1 / len(reverse_neigh[nb]) for nb in neighborhoods[i] if nb in uncovered)
            score = count * 1000 + rareness
            if score > best_score:
                best_score = score
                best_idx = i
        
        code_indices.append(best_idx)
        for nb in neighborhoods[best_idx]:
            if nb in uncovered:
                uncovered.discard(nb)
    
    # Phase 2: Remove redundant codewords
    code_set = set(code_indices)
    coverage = np.zeros(total_words, dtype=np.int32)
    for c in code_set:
        for nb in neighborhoods[c]:
            coverage[nb] += 1
    
    improved = True
    while improved:
        improved = False
        for c in list(code_set):
            removable = all(coverage[nb] > 1 for nb in neighborhoods[c])
            if removable:
                code_set.remove(c)
                for nb in neighborhoods[c]:
                    coverage[nb] -= 1
                improved = True
    
    best_code = list(code_set)
    best_size = len(best_code)
    
    # Phase 3: Local search using maximum matching
    # Key idea: Given current code set, build bipartite graph between 
    # "critical words" (covered exactly once) and their unique covering codewords.
    # If we can find a matching that shows some codewords are replaceable, we can swap.
    
    start_time = time.time()
    time_limit = 110  # seconds
    
    random.seed(42)
    
    def hopcroft_karp(left_nodes, right_nodes, adj):
        """
        Find maximum matching in bipartite graph.
        left_nodes: list of left-side node indices
        right_nodes: list of right-side node indices  
        adj: dict mapping left node -> list of right nodes
        Returns: matching dict {left: right}
        """
        # Map right nodes to 0-indexed
        right_index = {r: i for i, r in enumerate(right_nodes)}
        match_l = {}
        match_r = {}
        
        def bfs():
            dist = {}
            queue = deque()
            for u in left_nodes:
                if u not in match_l:
                    dist[u] = 0
                    queue.append(u)
            found = False
            while queue:
                u = queue.popleft()
                for v in adj.get(u, []):
                    if v not in right_index:
                        continue
                    vr = right_index[v]
                    if vr not in match_r:
                        found = True
                    elif match_r.get(vr) is not None and match_r[vr] not in dist:
                        dist[match_r[vr]] = dist[u] + 1
                        queue.append(match_r[vr])
            return found
        
        def dfs(u):
            for v in adj.get(u, []):
                if v not in right_index:
                    continue
                vr = right_index[v]
                if vr not in match_r:
                    match_l[u] = v
                    match_r[vr] = u
                    return True
                elif match_r.get(vr) is not None:
                    mu = match_r[vr]
                    if mu in dist and dist.get(mu, -1) == dist.get(u, -1) + 1:
                        # Actually need dist[mu] == dist[u] + 1
                        pass
            return False
        
        # Simplified HK implementation
        match_l = {}
        match_r = {}
        
        def bfs_build():
            dist = {}
            queue = deque()
            for u in left_nodes:
                if u not in match_l:
                    dist[u] = 0
                    queue.append(u)
            found = False
            while queue:
                u = queue.popleft()
                for v in adj.get(u, []):
                    vr = right_index.get(v)
                    if vr is None:
                        continue
                    if vr not in match_r:
                        found = True
                    else:
                        mu = match_r[vr]
                        if mu not in dist:
                            dist[mu] = dist[u] + 1
                            queue.append(mu)
            return found, dist
        
        def dfs_augment(u, dist, visited):
            for v in adj.get(u, []):
                vr = right_index.get(v)
                if vr is None:
                    continue
                if vr not in match_r:
                    match_l[u] = v
                    match_r[vr] = u
                    return True
                else:
                    mu = match_r[vr]
                    if mu not in visited and dist.get(mu, -1) == dist.get(u, -1) + 1:
                        visited.add(mu)
                        if dfs_augment(mu, dist, visited):
                            match_l[u] = v
                            match_r[vr] = u
                            return True
            return False
        
        while True:
            found, dist = bfs_build()
            if not found:
                break
            for u in left_nodes:
                if u not in match_l:
                    dfs_augment(u, dist, set())
        
        return match_l
    
    # Local search iterations
    no_improve_count = 0
    max_no_improve = 30
    
    while time.time() - start_time < time_limit and no_improve_count < max_no_improve:
        code_list = list(code_set)
        if len(code_list) == 0:
            break
            
        # Recompute coverage
        coverage = np.zeros(total_words, dtype=np.int32)
        for c in code_set:
            for nb in neighborhoods[c]:
                coverage[nb] += 1
        
        # Find critical words (covered exactly once)
        critical_words = [i for i in range(total_words) if coverage[i] == 1]
        
        if not critical_words:
            # No critical words, try random perturbation
            if len(code_set) > 1:
                c_remove = random.choice(list(code_set))
                code_set.remove(c_remove)
                for nb in neighborhoods[c_remove]:
                    coverage[nb] -= 1
                # Try to repair
                uncovered_after = [i for i in range(total_words) if coverage[i] == 0]
                if not uncovered_after:
                    best_size -= 1
                    best_code = list(code_set)
                    no_improve_count = 0
                    continue
                else:
                    # Revert
                    code_set.add(c_remove)
                    for nb in neighborhoods[c_remove]:
                        coverage[nb] += 1
            no_improve_count += 1
            continue
        
        # For each critical word, find its unique covering codeword
        word_to_coder = {}
        for w in critical_words:
            coders = [c for c in code_set if w in neighborhoods[c]]
            if len(coders) == 1:
                word_to_coder[w] = coders[0]
        
        # Group critical words by their covering codeword
        coder_to_words = defaultdict(list)
        for w, c in word_to_coder.items():
            coder_to_words[c].append(w)
        
        # Try to find alternative codewords that can cover these critical words
        # Build bipartite graph: left = codewords that cover critical words, right = critical words
        # We want to see if we can reassign coverage
        
        # For each codeword in code_set that covers some critical words,
        # see if we can replace it with a different codeword
        
        found_improvement = False
        
        # Strategy: pick a codeword that covers critical words, try to replace it
        # with a codeword not in the current set that covers those same critical words
        # and possibly more
        
        candidates_to_replace = [c for c in coder_to_words.keys() if len(coder_to_words[c]) >= 1]
        random.shuffle(candidates_to_replace)
        
        for c_out in candidates_to_replace[:20]:  # Try up to 20 candidates
            if time.time() - start_time > time_limit:
                break
            
            words_to_cover = coder_to_words[c_out]
            
            # Find codewords not in code_set that cover all words in words_to_cover
            # and check if they also cover other words to potentially allow further reduction
            best_c_in = None
            best_extra = -1
            
            # Sample some candidates
            sample_size = min(50, total_words - len(code_set))
            if sample_size <= 0:
                continue
                
            candidates = random.sample([i for i in indices if i not in code_set], sample_size)
            
            for c_in in candidates:
                # Check if c_in covers all words in words_to_cover
                covers_all = all(w in neighborhoods[c_in] for w in words_to_cover)
                if not covers_all:
                    continue
                
                # Count how many additional words c_in covers that are currently covered exactly once
                extra = 0
                for nb in neighborhoods[c_in]:
                    if coverage[nb] == 1 and nb not in words_to_cover:
                        # This word is covered by c_out, so if we remove c_out, it becomes uncovered
                        # unless c_in covers it
                        if nb in neighborhoods[c_out]:
                            extra += 1
                        else:
                            # c_in covers a word that's covered by another codeword
                            # This could help if that other codeword becomes redundant
                            extra += 0.5
                
                if extra > best_extra:
                    best_extra = extra
                    best_c_in = c_in
            
            if best_c_in is not None:
                # Perform the swap
                code_set.remove(c_out)
                code_set.add(best_c_in)
                
                # Update coverage
                for nb in neighborhoods[c_out]:
                    coverage[nb] -= 1
                for nb in neighborhoods[best_c_in]:
                    coverage[nb] += 1
                
                # Check validity
                if np.min(coverage) >= 1:
                    # Try to remove redundant codewords
                    removed_any = False
                    changed = True
                    while changed:
                        changed = False
                        for c in list(code_set):
                            removable = all(coverage[nb] > 1 for nb in neighborhoods[c])
                            if removable:
                                code_set.remove(c)
                                for nb in neighborhoods[c]:
                                    coverage[nb] -= 1
                                removed_any = True
                                changed = True
                                best_size -= 1
                                best_code = list(code_set)
                    
                    if removed_any or len(code_set) < best_size:
                        found_improvement = True
                        no_improve_count = 0
                        break
                    else:
                        # Revert
                        code_set.remove(best_c_in)
                        code_set.add(c_out)
                        for nb in neighborhoods[best_c_in]:
                            coverage[nb] -= 1
                        for nb in neighborhoods[c_out]:
                            coverage[nb] += 1
                else:
                    # Invalid, revert
                    code_set.remove(best_c_in)
                    code_set.add(c_out)
                    for nb in neighborhoods[best_c_in]:
                        coverage[nb] -= 1
                    for nb in neighborhoods[c_out]:
                        coverage[nb] += 1
        
        if not found_improvement:
            no_improve_count += 1
    
    return [list(idx_to_word(i)) for i in best_code]

# EVOLVE-BLOCK-END
