import time
from itertools import combinations

def construct():
    """
    Returns the 132 blocks of the Steiner System S(5, 6, 12).
    Uses Algorithm X (Dancing Links) to find the exact cover of the 792 5-subsets
    by the 924 6-blocks.
    """
    start_time = time.time()
    deadline = start_time + 145.0
    
    # 1. Precompute 5-subsets and 6-subsets
    points = range(12)
    all_5subsets = list(combinations(points, 5))
    all_6subsets = list(combinations(points, 6))
    
    n_5 = len(all_5subsets) # 792
    n_6 = len(all_6subsets) # 924
    
    # Map 5-subsets to column indices
    five_to_col = {sub: i for i, sub in enumerate(all_5subsets)}
    
    # Build the matrix: list of lists of column indices for each row (block)
    # row i corresponds to all_6subsets[i]
    matrix = []
    for block in all_6subsets:
        cols = []
        for i in range(6):
            sub = block[:i] + block[i+1:]
            cols.append(five_to_col[sub])
        matrix.append(cols)
        
    # 2. Implement Dancing Links (DLX)
    # We use arrays for L, R, U, D, Col, Row to simulate the circular doubly linked lists.
    # Size: max_nodes = n_5 + 1 (header) + n_6 * 6 (entries)
    max_nodes = n_5 + 1 + n_6 * 6
    L = [0] * max_nodes
    R = [0] * max_nodes
    U = [0] * max_nodes
    D = [0] * max_nodes
    Col = [0] * max_nodes
    Row = [0] * max_nodes
    size = [0] * (n_5 + 1) # size of column j
    header = 0
    node_count = 1
    
    # Initialize header row
    for j in range(n_5 + 1):
        R[j] = (j + 1) % (n_5 + 1)
        L[j] = (j - 1) % (n_5 + 1)
        D[j] = j
        U[j] = j
        size[j] = 0
        
    # Add rows to the matrix
    for i in range(n_6):
        cols = matrix[i]
        first_node = node_count
        for c in cols:
            # Create new node
            node = node_count
            node_count += 1
            
            # Insert node into column c
            U[node] = U[c]
            D[node] = c
            D[U[c]] = node
            U[c] = node
            size[c] += 1
            
            # Link horizontally
            if node == first_node:
                L[node] = node
                R[node] = node
            else:
                # Insert after the previous node in this row
                prev_node = node - 1
                R[node] = R[prev_node]
                L[node] = prev_node
                L[R[prev_node]] = node
                R[prev_node] = node
                
            Col[node] = c
            Row[node] = i
            
    # DLX Search
    solution = []
    
    def remove(node):
        R[L[node]] = R[node]
        L[R[node]] = L[node]
        
    def restore(node):
        R[L[node]] = node
        L[R[node]] = node
        
    def cover(col):
        remove(col)
        i = D[col]
        while i != col:
            j = R[i]
            while j != i:
                remove(j)
                j = R[j]
            i = D[i]
            
    def uncover(col):
        i = U[col]
        while i != col:
            j = L[i]
            while j != i:
                restore(j)
                j = L[j]
            i = U[i]
        restore(col)
        
    def search(k):
        if time.time() > deadline:
            return False
            
        if R[header] == header:
            # All columns covered
            return True
            
        # Choose column with minimum size (MRV)
        col = R[header]
        min_size = size[col]
        j = R[col]
        while j != header:
            if size[j] < min_size:
                min_size = size[j]
                col = j
            j = R[j]
            
        cover(col)
        r = D[col]
        while r != col:
            # Add row r to solution
            solution.append(Row[r])
            
            # Cover all other columns in this row
            j = R[r]
            while j != r:
                cover(Col[j])
                j = R[j]
                
            if search(k + 1):
                return True
                
            # Undo
            j = L[r]
            while j != r:
                uncover(Col[j])
                j = L[j]
                
            solution.pop()
            r = D[r]
            
        uncover(col)
        return False
        
    # We know the solution has 132 blocks.
    # We can just run search and stop when we find a solution.
    # Since the solution is unique (up to isomorphism), finding one is sufficient.
    
    search(0)
    
    # Extract the blocks
    blocks = [list(all_6subsets[idx]) for idx in solution]
    
    return blocks

if __name__ == '__main__':
    res = construct()
    print(f"Number of blocks: {len(res)}")
