import time
import sys
import itertools

def construct(n):
    if n <= 1:
        return "1" if n == 1 else ""
    if n == 2:
        return "121"
    if n == 3:
        return "1231321"
    if n == 4:
        return "123413241423124313421"
    
    start_time = time.time()
    deadline = start_time + 130  # 130s limit, 20s safety margin
    
    # Cycle decomposition approach for n >= 5
    # Nodes are (n-1)-tuples, edges determined by missing symbol
    
    def get_next_node(node):
        present = set(node)
        for i in range(1, n + 1):
            if i not in present:
                return node[1:] + (i,)
        return None
    
    # Find all cycles
    visited = set()
    cycles = []
    
    # Iterate over all nodes
    for missing in range(1, n + 1):
        remaining = [x for x in range(1, n + 1) if x != missing]
        for perm in itertools.permutations(remaining):
            if perm in visited:
                continue
            
            cycle = []
            curr = perm
            while curr not in visited:
                visited.add(curr)
                cycle.append(curr)
                curr = get_next_node(curr)
            
            if cycle:
                cycles.append(cycle)
            
            if time.time() > deadline:
                break
    
    # If we found cycles, try to order them optimally
    if cycles and time.time() < deadline - 30:
        # Each cycle has a "start" node (first element) and "end" node (last element)
        # The end node's next is the start node
        # 
        # To concatenate, we need to find the cost of going from end of one cycle
        # to start of another.
        #
        # Actually, let me think about the word structure.
        # A path v0 -> v1 -> ... -> vk gives word: v0[0]v0[1]...v0[n-2]v1[n-2]...vk[n-2]
        #
        # For a cycle v0 -> v1 -> ... -> v_{m-1} -> v0:
        # Word: v0[0]v0[1]...v0[n-2] v1[n-2] v2[n-2] ... v_{m-1}[n-2]
        # Length: (n-1) + m
        #
        # When concatenating cycle A (ending at node a_last, where get_next_node(a_last) = a_first)
        # and cycle B (starting at node b_first):
        # We need to append symbols to go from a_last to b_first.
        #
        # The current word ends with ...a_last[n-2]
        # We need to reach b_first.
        #
        # Let's compute the minimum additional symbols needed.
        # Start from a_last, follow the path until we reach b_first or any node in cycle B.
        #
        # This is getting complex. Let me use a simpler heuristic:
        # Order cycles by their length (shortest first) and just concatenate with minimal overlap.
        
        # Sort cycles by length
        cycles.sort(key=len)
        
        # Build the word
        word = []
        
        for i, cycle in enumerate(cycles):
            if i == 0:
                # First cycle: write out the full word
                v0 = cycle[0]
                word.extend(list(v0))
                for j in range(1, len(cycle)):
                    word.append(cycle[j][n-2])
            else:
                # Transition from previous cycle to this one
                # The previous cycle ended at some node, and we need to reach cycle[i][0]
                # 
                # For now, just append the full cycle with minimal overlap
                v0 = cycle[0]
                block = list(v0)
                for j in range(1, len(cycle)):
                    block.append(cycle[j][n-2])
                
                # Find maximum overlap with current word
                max_overlap = min(len(word), len(block) - 1)
                overlap = 0
                for k in range(max_overlap, -1, -1):
                    if k == 0:
                        break
                    if word[-k:] == block[:k]:
                        overlap = k
                        break
                word.extend(block[overlap:])
        
        result = "".join(map(str, word))
        
        # Verify it's valid
        if time.time() < deadline - 10:
            seen_perms = set()
            for i in range(len(result) - n + 1):
                perm = tuple(int(c) for c in result[i:i+n])
                if len(set(perm)) == n:
                    seen_perms.add(perm)
            
            if len(seen_perms) == int(itertools.permutations(range(1, n+1).__len__()) if False else 1):
                pass  # This check is wrong, let me fix it
            
            # Proper check
            expected = 1
            for i in range(2, n+1):
                expected *= i
            
            if len(seen_perms) == expected:
                return result
    
    # Fallback: standard recursive construction
    S = [1]
    for m in range(2, n + 1):
        seen = set()
        perms_order = []
        for i in range(len(S) - (m - 1) + 1):
            window = tuple(S[i:i + m - 1])
            if window not in seen:
                seen.add(window)
                perms_order.append(window)
        
        new_S = []
        for P in perms_order:
            block = list(P) + [m] + list(P)
            max_possible_overlap = min(len(new_S), len(block) - 1)
            overlap = 0
            for k in range(max_possible_overlap, -1, -1):
                if k == 0:
                    break
                if new_S[-k:] == block[:k]:
                    overlap = k
                    break
            new_S.extend(block[overlap:])
        S = new_S
        
        if time.time() > deadline:
            break
    
    return "".join(map(str, S))
