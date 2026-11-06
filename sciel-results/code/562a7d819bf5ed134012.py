import time
from itertools import combinations

class DLXNode:
    __slots__ = ['L', 'R', 'U', 'D', 'C', 'row', 'col']
    def __init__(self):
        self.L = None
        self.R = None
        self.U = None
        self.D = None
        self.C = None
        self.row = -1
        self.col = -1

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # 1. Define the exact cover problem
    # Points: 0..11
    # Columns: All 5-subsets of 12 points (792 columns)
    # Rows: All 6-subsets of 12 points (924 rows)
    # Each row (6-subset) covers 6 columns (its 5-subsets)
    
    points = list(range(12))
    all_5_subsets = list(combinations(points, 5))
    col_index = {s: i for i, s in enumerate(all_5_subsets)}
    num_cols = len(all_5_subsets) # 792
    
    all_6_subsets = list(combinations(points, 6))
    num_rows = len(all_6_subsets) # 924
    
    # 2. Build the DLX matrix
    # Header row
    header = [DLXNode() for _ in range(num_cols + 1)]
    for i in range(num_cols + 1):
        header[i].L = header[i-1] if i > 0 else header[-1]
        header[i].R = header[i+1] if i < num_cols else header[0]
        header[i].C = header[i]
        header[i].U = header[i]
        header[i].D = header[i]
        header[i].col = i
        header[i].row = -1
    
    # Column headers are already linked in a circle
    
    # Add rows
    # We need to store which row corresponds to which 6-subset
    row_to_block = {}
    
    for r_idx, block in enumerate(all_6_subsets):
        # Find the columns covered by this block
        covered_cols = []
        for c in combinations(block, 5):
            covered_cols.append(col_index[c])
        
        # Create nodes for this row
        first_node = None
        for c_idx in covered_cols:
            node = DLXNode()
            node.row = r_idx
            node.col = c_idx
            node.C = header[c_idx]
            
            # Insert into column
            node.U = header[c_idx].U
            node.D = header[c_idx]
            header[c_idx].U.D = node
            header[c_idx].U = node
            
            # Link to next node in row
            if first_node is None:
                first_node = node
                node.L = node
                node.R = node
            else:
                node.L = first_node.L
                node.R = first_node
                first_node.L.R = node
                first_node.L = node
            
            # Track nodes for this row
            if not hasattr(construct, '_nodes'):
                construct._nodes = []
            construct._nodes.append(node)
            
        row_to_block[r_idx] = list(block)
    
    # 3. DLX Search
    solution_rows = []
    
    def cover(c):
        c.R.L = c.L
        c.L.R = c.R
        i = c.D
        while i != c:
            j = i.R
            while j != i:
                j.U.D = j.D
                j.D.U = j.U
                j.C.size -= 1
                j = j.R
            i = i.D

    def uncover(c):
        i = c.U
        while i != c:
            j = i.L
            while j != i:
                j.C.size += 1
                j.U.D = j
                j.D.U = j
                j = j.L
            i = i.U
        c.R.L = c
        c.L.R = c

    # Add size attribute to headers
    for i in range(num_cols + 1):
        header[i].size = 0
        # Recount size
        # Actually, let's just recount at the end of construction or initialize properly
    # Re-initialize sizes
    for i in range(1, num_cols + 1):
        header[i].size = 0
    for r_idx in range(num_rows):
        for c_idx in combinations(row_to_block[r_idx], 5):
            header[col_index[c_idx]].size += 1

    def search(k):
        if time.time() > deadline:
            return False
        if header[0].R == header[0]:
            return True
        
        # Choose column with minimum size (heuristic)
        c = header[0].R
        while c != header[0]:
            if c.size < c.R.size:
                c = c.R
            else:
                c = c.R
            if c.size == 0:
                return False
        
        cover(c)
        r = c.D
        while r != c:
            solution_rows.append(r.row)
            j = r.R
            while j != r:
                cover(j.C)
                j = j.R
            if search(k + 1):
                return True
            j = r.L
            while j != r:
                uncover(j.C)
                j = j.L
            solution_rows.pop()
            r = r.D
        uncover(c)
        return False
    
    # Start search
    found = search(0)
    
    if found:
        blocks = []
        for r_idx in solution_rows:
            blocks.append(row_to_block[r_idx])
        
        # Verify
        if len(blocks) == 132:
            covered = set()
            valid = True
            for b in blocks:
                for s in combinations(b, 5):
                    if s in covered:
                        valid = False
                        break
                    covered.add(s)
                if not valid:
                    break
            if valid and len(covered) == 792:
                return blocks
    
    # Fallback: Greedy
    covered = set()
    greedy_blocks = []
    for b in all_6_subsets:
        subsets = list(combinations(b, 5))
        if all(s not in covered for s in subsets):
            greedy_blocks.append(list(b))
            covered.update(subsets)
    
    return greedy_blocks
