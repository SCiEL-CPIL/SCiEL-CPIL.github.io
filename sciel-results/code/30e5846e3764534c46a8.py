import itertools
import time
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
        
    # 1. Generate all permutations of 1..n
    perms = list(itertools.permutations(range(1, n + 1)))
    
    # 2. Group permutations into rotation classes of their (n-1)-prefixes?
    # No, the blocks are based on (n-1)-perms.
    # Wait, the standard construction inserts n into (n-1)-perms.
    # My "rotation class" idea was based on the (n-1)-perms P.
    # The blocks are B_P where P is a perm of 1..n-1.
    # There are (n-1)! such blocks.
    # For n=7, there are 720 blocks.
    # Rotation classes of 6-perms: 720 / 6 = 120 classes.
    
    # So we construct the superpermutation for n by ordering the blocks B_P
    # where P is a permutation of 1..n-1.
    
    # Generate all (n-1) permutations
    base_perms = list(itertools.permutations(range(1, n)))
    
    # Group base_perms into rotation classes
    # Canonical form: tuple of the lexicographically smallest rotation
    def canonical_rot(p):
        min_rot = p
        for i in range(1, len(p)):
            rot = p[i:] + p[:i]
            if rot < min_rot:
                min_rot = rot
        return min_rot

    classes = defaultdict(list)
    for p in base_perms:
        classes[canonical_rot(p)].append(p)
        
    # For each class, order the perms in rotation order
    # To maximize internal overlap (n-2)
    class_blocks = []
    for can, members in classes.items():
        # Sort members to form a rotation cycle
        # Start with can, then rotate
        # Ensure we visit all members
        # Since it's a cycle, we can just sort by their position in the cycle
        # Easier: just sort them arbitrarily? No, we need the rotation order to get n-2 overlap.
        # Let's find the cycle order.
        # Pick an element, say can.
        # Next is rotate(can, 1).
        # ...
        ordered = []
        current = can
        for _ in range(len(members)):
            ordered.append(current)
            # Rotate left
            current = current[1:] + current[:1]
        # Verify all are in members
        if set(ordered) != set(members):
            # Fallback to arbitrary order if something went wrong
            ordered = sorted(members)
        class_blocks.append(ordered)
        
    # Now we have ~120 classes (for n=7).
    # We need to order these classes to maximize overlap between the last block of one class
    # and the first block of the next class.
    
    # Precompute blocks for each class start and end
    # Block for perm P: list(P) + [n] + list(P)
    def get_block(p):
        return list(p) + [n] + list(p)
        
    # For each class, store the sequence of blocks
    # And the "end" of the class (last block) and "start" of the class (first block)
    # Actually, we can choose the start of the class to optimize the transition?
    # Yes! The class is a cycle. We can start at any node in the cycle.
    # So for each class, we have (n-1) possible starts.
    # This increases the TSP nodes from 120 to 120 * 6 = 720?
    # No, we can just pick the best start after ordering the classes?
    # Or include start choice in the TSP?
    # Let's keep it simple: Fix the start of each class to be the canonical form.
    # If we want to optimize further, we can try all starts.
    
    class_data = []
    for ordered in class_blocks:
        blocks = [get_block(p) for p in ordered]
        class_data.append({
            'blocks': blocks,
            'start': blocks[0],
            'end': blocks[-1]
        })
        
    # Greedy TSP on classes
    # Number of classes M
    M = len(class_data)
    
    # Precompute overlap between end of class i and start of class j
    # overlap(end_i, start_j)
    # end_i is a list of ints. start_j is a list of ints.
    # Max overlap k such that end_i[-k:] == start_j[:k]
    
    def calc_overlap(s1, s2):
        max_k = min(len(s1), len(s2))
        for k in range(max_k, 0, -1):
            if s1[-k:] == s2[:k]:
                return k
        return 0
        
    # M is 120 for n=7.
    # 120x120 matrix.
    overlap_mat = [[0]*M for _ in range(M)]
    for i in range(M):
        for j in range(M):
            if i != j:
                overlap_mat[i][j] = calc_overlap(class_data[i]['end'], class_data[j]['start'])
                
    # Greedy Nearest Neighbor
    visited = [False] * M
    path = []
    current = 0
    visited[0] = True
    path.append(0)
    
    for _ in range(M - 1):
        best_next = -1
        best_overlap = -1
        for j in range(M):
            if not visited[j]:
                if overlap_mat[current][j] > best_overlap:
                    best_overlap = overlap_mat[current][j]
                    best_next = j
        if best_next == -1:
            # Find any unvisited
            for j in range(M):
                if not visited[j]:
                    best_next = j
                    break
        visited[best_next] = True
        path.append(best_next)
        current = best_next
        
    # Construct the word
    word = []
    for idx in path:
        blocks = class_data[idx]['blocks']
        for b in blocks:
            # Overlap with current word
            max_k = min(len(word), len(b))
            overlap = 0
            for k in range(max_k, 0, -1):
                if word[-k:] == b[:k]:
                    overlap = k
                    break
            word.extend(b[overlap:])
            
    # Trim ends
    # Calculate frequency
    freq = {}
    L = len(word)
    for i in range(L - n + 1):
        p = tuple(word[i:i+n])
        freq[p] = freq.get(p, 0) + 1
        
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        if len(word) > n:
            first_perm = tuple(word[:n])
            if freq.get(first_perm, 0) > 1:
                freq[first_perm] -= 1
                word = word[1:]
                changed = True
        if len(word) > n:
            last_perm = tuple(word[-n:])
            if freq.get(last_perm, 0) > 1:
                freq[last_perm] -= 1
                word = word[:-1]
                changed = True
                
    return "".join(map(str, word))
