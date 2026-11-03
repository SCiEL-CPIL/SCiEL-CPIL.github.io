import time
import itertools
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    # For the specific case q=3, n=7, radius=1, we can use a faster approach.
    # General case: greedy set cover with local search improvement.
    
    # Precompute all words
    all_words = list(itertools.product(range(q), repeat=n))
    num_words = len(all_words)
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors for each word (words within Hamming distance <= radius)
    # For radius=1, this is fast. For larger radius, use BFS.
    neighbors = [[] for _ in range(num_words)]
    
    if radius == 1:
        # Fast path for radius=1
        for i, w in enumerate(all_words):
            neighbors[i].append(i)
            for pos in range(n):
                orig = w[pos]
                for val in range(q):
                    if val == orig:
                        continue
                    nw = list(w)
                    nw[pos] = val
                    idx = word_to_idx[tuple(nw)]
                    neighbors[i].append(idx)
    else:
        # General BFS for neighbors
        for i, w in enumerate(all_words):
            visited = {i}
            queue = [(i, 0)]
            while queue:
                curr, dist = queue.pop(0)
                neighbors[i].append(curr)
                if dist == radius:
                    continue
                t = list(all_words[curr])
                for pos in range(n):
                    orig = t[pos]
                    for val in range(q):
                        if val == orig:
                            continue
                        t[pos] = val
                        nw = tuple(t)
                        idx = word_to_idx[nw]
                        if idx not in visited:
                            visited.add(idx)
                            queue.append((idx, dist + 1))
                    t[pos] = orig
    
    # Greedy set cover
    uncovered = set(range(num_words))
    code = []
    
    while uncovered:
        if time.time() > deadline - 10:
            break
            
        best_w = -1
        best_count = 0
        
        for w_idx in range(num_words):
            count = 0
            for n_idx in neighbors[w_idx]:
                if n_idx in uncovered:
                    count += 1
            if count > best_count:
                best_count = count
                best_w = w_idx
        
        if best_w == -1:
            break
            
        code.append(all_words[best_w])
        
        for n_idx in neighbors[best_w]:
            if n_idx in uncovered:
                uncovered.remove(n_idx)
    
    # If we couldn't finish greedy due to time, fallback to a valid code
    if uncovered:
        # Return all words as a valid but large code
        return all_words
    
    # Local search to improve the solution
    # Try to remove or swap codewords to reduce the count while maintaining coverage
    
    selected_idx = [word_to_idx[w] for w in code]
    selected_set = set(selected_idx)
    
    # Compute coverage: for each word, which selected codewords cover it
    # covered_by[word_idx] = list of selected_idx that cover word_idx
    
    # First, let's build a coverage matrix or at least know which words are covered by how many
    # For each word in the space, count how many selected codewords cover it
    
    coverage_count = [0] * num_words
    for s_idx in selected_idx:
        for n_idx in neighbors[s_idx]:
            coverage_count[n_idx] += 1
    
    # Try to remove a codeword if all words it covers are covered by at least one other
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i, s_idx in enumerate(selected_idx):
            # Check if all words covered by s_idx are covered by another selected codeword
            removable = True
            for n_idx in neighbors[s_idx]:
                # Count how many selected codewords cover n_idx, excluding s_idx
                cnt = 0
                for j, other_idx in enumerate(selected_idx):
                    if j == i:
                        continue
                    if n_idx in neighbors[other_idx]:
                        cnt += 1
                        if cnt >= 1:
                            break
                if cnt == 0:
                    removable = False
                    break
            
            if removable:
                # Remove s_idx
                selected_idx.pop(i)
                improved = True
                # Recompute coverage_count
                coverage_count = [0] * num_words
                for s_idx2 in selected_idx:
                    for n_idx2 in neighbors[s_idx2]:
                        coverage_count[n_idx2] += 1
                break  # Restart the loop after a removal
    
    # Try swaps: replace one codeword with another to potentially allow further removals
    # This is a more complex local search. For now, let's try a simple version.
    # Try to find a codeword that, when added, allows removal of two or more existing codewords.
    # This is expensive, so we'll do a limited version.
    
    if time.time() < deadline - 10:
        # Try adding a random codeword and see if we can remove multiple
        random.seed(42)
        for _ in range(200):
            if time.time() > deadline - 5:
                break
            
            # Pick a random codeword not in the current selection
            candidates = []
            for w_idx in range(num_words):
                if w_idx not in selected_set:
                    candidates.append(w_idx)
            
            if not candidates:
                break
            
            new_idx = random.choice(candidates)
            new_selected = selected_idx + [new_idx]
            
            # Try to greedily remove codewords from new_selected
            new_selected_set = set(new_selected)
            new_cov = [0] * num_words
            for s_idx in new_selected:
                for n_idx in neighbors[s_idx]:
                    new_cov[n_idx] += 1
            
            removable_indices = []
            for i, s_idx in enumerate(new_selected):
                removable = True
                for n_idx in neighbors[s_idx]:
                    cnt = 0
                    for j, other_idx in enumerate(new_selected):
                        if j == i:
                            continue
                        if n_idx in neighbors[other_idx]:
                            cnt += 1
                            if cnt >= 1:
                                break
                    if cnt == 0:
                        removable = False
                        break
                if removable:
                    removable_indices.append(i)
            
            if len(removable_indices) >= 2:
                # Try removing the first two removable
                new_selected2 = [s for i, s in enumerate(new_selected) if i not in removable_indices[:2]]
                if len(new_selected2) < len(selected_idx):
                    selected_idx = new_selected2
                    selected_set = set(selected_idx)
            
            selected_set = set(selected_idx)
    
    # Convert back to list of codewords
    code = [all_words[s_idx] for s_idx in selected_idx]
    
    return code
