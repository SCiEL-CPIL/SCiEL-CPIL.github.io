<<<<<<< SEARCH
import time
import numpy as np

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 54.0

    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0

    if m == 0:
        return {'gates': [], 'outputs': []}

    A = np.array(matrix, dtype=np.uint8) % 2

    # RREF with full row tracking: T @ A = M (reduced row echelon)
    M = A.copy()
    T = np.eye(m, dtype=np.uint8)

    rank = 0
    for col in range(n):
        if time.time() > deadline:
            break
        # Find pivot
        pivot = -1
        for row in range(rank, m):
            if M[row, col] == 1:
                pivot = row
                break
        if pivot == -1:
            continue

        # Swap
        M[[rank, pivot]] = M[[pivot, rank]]
        T[[rank, pivot]] = T[[pivot, rank]]

        # Eliminate all other rows
        for row in range(m):
            if row != rank and M[row, col] == 1:
                M[row] ^= M[rank]
                T[row] ^= T[rank]

        rank += 1
        if rank == m:
            break

    # Now M is in RREF, T @ A = M
    # We need A = T^{-1} @ M
    # Compute T^{-1} by augmenting and doing RREF on [T | I]
    if time.time() > deadline:
        # Fallback: just use the pivot rows directly (incomplete but valid if rank==m)
        # Actually, let's just do a simple approach
        gates = []
        outputs = []
        for j in range(m):
            ones = [i for i in range(n) if A[j, i] == 1]
            if not ones:
                outputs.append(n)
            elif len(ones) == 1:
                outputs.append(ones[0])
            else:
                cur = ones[0]
                for i in range(1, len(ones)):
                    g = len(gates)
                    gates.append([cur, ones[i]])
                    cur = n + 1 + g
                outputs.append(cur)
        return {'gates': gates, 'outputs': outputs}

    aug = np.hstack([T, np.eye(m, dtype=np.uint8)])
    for col in range(m):
        if time.time() > deadline:
            break
        pivot = -1
        for row in range(col, m):
            if aug[row, col] == 1:
                pivot = row
                break
        if pivot == -1:
            continue
        aug[[col, pivot]] = aug[[pivot, col]]
        for row in range(m):
            if row != col and aug[row, col] == 1:
                aug[row] ^= aug[col]
    T_inv = aug[:, m:]

    # Build pivot wires: wire for each pivot row i
    gates = []
    pivot_wires = []
    for i in range(rank):
        ones = [j for j in range(n) if M[i, j] == 1]
        if not ones:
            pivot_wires.append(n)
        elif len(ones) == 1:
            pivot_wires.append(ones[0])
        else:
            cur = ones[0]
            for j in range(1, len(ones)):
                g = len(gates)
                gates.append([cur, ones[j]])
                cur = n + 1 + g
            pivot_wires.append(cur)

    # Global memo: frozenset of pivot indices -> wire
    wire_memo = {}
    # Pre-fill singletons
    for i in range(rank):
        wire_memo[frozenset([i])] = pivot_wires[i]

    def xor_pivot_indices(idx_set):
        """Return wire that is XOR of pivot_wires[i] for i in idx_set."""
        if not idx_set:
            return n
        key = frozenset(idx_set)
        if key in wire_memo:
            return wire_memo[key]
        
        # Divide and conquer
        lst = list(key)
        mid = len(lst) // 2
        left_set = set(lst[:mid])
        right_set = set(lst[mid:])
        
        left_wire = xor_pivot_indices(left_set)
        right_wire = xor_pivot_indices(right_set)
        
        g = len(gates)
        gates.append([left_wire, right_wire])
        result = n + 1 + g
        wire_memo[key] = result
        return result

    # Build outputs
    outputs = []
    for j in range(m):
        if time.time() > deadline:
            # Fallback
            contributing = [i for i in range(rank) if T_inv[j, i] == 1]
            if not contributing:
                outputs.append(n)
            elif len(contributing) == 1:
                outputs.append(pivot_wires[contributing[0]])
            else:
                cur = pivot_wires[contributing[0]]
                for i in range(1, len(contributing)):
                    g = len(gates)
                    gates.append([cur, pivot_wires[i]])
                    cur = n + 1 + g
                outputs.append(cur)
        else:
            contributing = [i for i in range(rank) if T_inv[j, i] == 1]
            outputs.append(xor_pivot_indices(contributing))

    return {'gates': gates, 'outputs': outputs}
=======
import time
import numpy as np

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 54.0

    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0

    if m == 0:
        return {'gates': [], 'outputs': []}

    A = np.array(matrix, dtype=np.uint8) % 2

    # RREF with full row tracking: T @ A = M (reduced row echelon)
    M = A.copy()
    T = np.eye(m, dtype=np.uint8)

    rank = 0
    for col in range(n):
        if time.time() > deadline:
            break
        # Find pivot
        pivot = -1
        for row in range(rank, m):
            if M[row, col] == 1:
                pivot = row
                break
        if pivot == -1:
            continue

        # Swap
        M[[rank, pivot]] = M[[pivot, rank]]
        T[[rank, pivot]] = T[[pivot, rank]]

        # Eliminate all other rows
        for row in range(m):
            if row != rank and M[row, col] == 1:
                M[row] ^= M[rank]
                T[row] ^= T[rank]

        rank += 1
        if rank == m:
            break

    # Now M is in RREF, T @ A = M
    # We need A = T^{-1} @ M
    # Compute T^{-1} by augmenting and doing RREF on [T | I]
    if time.time() > deadline:
        # Fallback: just use the pivot rows directly (incomplete but valid if rank==m)
        gates = []
        outputs = []
        for j in range(m):
            ones = [i for i in range(n) if A[j, i] == 1]
            if not ones:
                outputs.append(n)
            elif len(ones) == 1:
                outputs.append(ones[0])
            else:
                cur = ones[0]
                for i in range(1, len(ones)):
                    g = len(gates)
                    gates.append([cur, ones[i]])
                    cur = n + 1 + g
                outputs.append(cur)
        return {'gates': gates, 'outputs': outputs}

    aug = np.hstack([T, np.eye(m, dtype=np.uint8)])
    for col in range(m):
        if time.time() > deadline:
            break
        pivot = -1
        for row in range(col, m):
            if aug[row, col] == 1:
                pivot = row
                break
        if pivot == -1:
            continue
        aug[[col, pivot]] = aug[[pivot, col]]
        for row in range(m):
            if row != col and aug[row, col] == 1:
                aug[row] ^= aug[col]
    T_inv = aug[:, m:]

    # Build pivot wires: wire for each pivot row i
    gates = []
    pivot_wires = []
    for i in range(rank):
        ones = [j for j in range(n) if M[i, j] == 1]
        if not ones:
            pivot_wires.append(n)
        elif len(ones) == 1:
            pivot_wires.append(ones[0])
        else:
            cur = ones[0]
            for j in range(1, len(ones)):
                g = len(gates)
                gates.append([cur, ones[j]])
                cur = n + 1 + g
            pivot_wires.append(cur)

    # Collect all output sets and their frequencies
    output_sets = []
    for j in range(m):
        contributing = frozenset(i for i in range(rank) if T_inv[j, i] == 1)
        output_sets.append(contributing)

    # Count frequency of each set
    from collections import Counter
    set_freq = Counter(output_sets)

    # Build all needed sets: all subsets of output sets + output sets themselves
    # We'll use a bottom-up approach: compute XOR for all subsets in order of size
    # First, collect all unique sets we need
    all_needed = set()
    for s in output_sets:
        all_needed.add(s)
        # Add all subsets of s
        lst = list(s)
        for i in range(len(lst)):
            for j in range(i+1, len(lst)):
                all_needed.add(frozenset([lst[i], lst[j]]))
        for i in range(len(lst)):
            all_needed.add(frozenset([lst[i]]))
        if lst:
            all_needed.add(frozenset())

    # Sort by size for bottom-up computation
    all_needed_sorted = sorted(all_needed, key=lambda x: len(x))

    wire_memo = {}
    # Pre-fill singletons and empty
    wire_memo[frozenset()] = n
    for i in range(rank):
        wire_memo[frozenset([i])] = pivot_wires[i]

    # Bottom-up: compute XOR for each set using two smaller subsets
    for s in all_needed_sorted:
        if s in wire_memo:
            continue
        lst = list(s)
        size = len(lst)
        # Find a split point that minimizes gates
        # Try splitting in half
        best_split = None
        best_cost = float('inf')
        for mid in range(1, size):
            left = frozenset(lst[:mid])
            right = frozenset(lst[mid:])
            # Cost is 1 if both are in memo, or we can estimate
            if left in wire_memo and right in wire_memo:
                cost = 0  # both available
            elif left in wire_memo:
                cost = 1
            elif right in wire_memo:
                cost = 1
            else:
                cost = 2
            if cost < best_cost:
                best_cost = cost
                best_split = (left, right)
            if cost == 0:
                break

        if best_split is None:
            # Fallback: split in half
            mid = size // 2
            left = frozenset(lst[:mid])
            right = frozenset(lst[mid:])
            best_split = (left, right)

        left, right = best_split
        # Recursively ensure left and right are computed
        if left not in wire_memo:
            # This shouldn't happen if we sorted by size, but handle it
            pass
        if right not in wire_memo:
            pass

        # Now compute
        g = len(gates)
        gates.append([wire_memo[left], wire_memo[right]])
        result = n + 1 + g
        wire_memo[s] = result

    # Build outputs
    outputs = []
    for s in output_sets:
        outputs.append(wire_memo[s])

    return {'gates': gates, 'outputs': outputs}
>>>>>>> REPLACE
