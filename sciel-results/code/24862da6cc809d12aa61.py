"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()

def _add_form(a, b):
    result = dict(a)
    for k, v in b.items():
        result[k] = result.get(k, 0) + v
    # Remove zero entries
    return {k: v for k, v in result.items() if v != 0}

def _sub_form(a, b):
    result = dict(a)
    for k, v in b.items():
        result[k] = result.get(k, 0) - v
    return {k: v for k, v in result.items() if v != 0}

def _strassen_2x2(Af, Bf, C_idx):
    """
    Af: dict {(0,0): form, (0,1): form, (1,0): form, (1,1): form} - A linear forms
    Bf: dict {(0,0): form, (0,1): form, (1,0): form, (1,1): form} - B linear forms
    C_idx: function (i,k) -> flat C index
    Returns list of 7 (A_form, B_form, C_dict) tuples
    """
    A00, A01 = Af[(0,0)], Af[(0,1)]
    A10, A11 = Af[(1,0)], Af[(1,1)]
    B00, B01 = Bf[(0,0)], Bf[(0,1)]
    B10, B11 = Bf[(1,0)], Bf[(1,1)]
    
    prods = []
    
    # Q1 = (A00+A11)(B00+B11)
    prods.append((_add_form(A00, A11), _add_form(B00, B11), {C_idx(0,0): 1, C_idx(1,1): 1}))
    # Q2 = (A10+A11)B00
    prods.append((_add_form(A10, A11), B00, {C_idx(1,0): 1}))
    # Q3 = A00(B10-B11)
    prods.append((A00, _sub_form(B10, B11), {C_idx(1,1): -1}))
    # Q4 = A11(B01-B10)
    prods.append((A11, _sub_form(B01, B10), {C_idx(0,1): 1, C_idx(1,1): 1}))
    # Q5 = (A00+A10)B11
    prods.append((_add_form(A00, A10), B11, {C_idx(0,0): -1, C_idx(1,0): 1}))
    # Q6 = (B00+B10)(A10-A11)
    prods.append((_sub_form(A10, A11), _add_form(B00, B10), {C_idx(0,0): 1}))
    # Q7 = (B01+B11)(A00-A10)
    prods.append((_sub_form(A00, A10), _add_form(B01, B11), {C_idx(0,0): 1, C_idx(0,1): 1}))
    
    return prods

def _build_4x4x4(b_col_offset, c_col_offset):
    """
    Build 49 products for 4×4×4 matrix multiplication.
    b_col_offset: offset for B column index (0 for first block, 4 for second)
    c_col_offset: offset for C column index (0 for first block, 4 for second)
    Returns list of 49 (A_form, B_form, C_dict) tuples.
    A_form keys: i*4+j (flat index in A, 0..15)
    B_form keys: j*9+(k+b_col_offset) (flat index in B, 0..35)
    C_dict keys: (k+c_col_offset)*4+i (flat index in C, 0..35)
    """
    # Define A block forms (2×2 blocks of the 4×4 A matrix)
    # A00 block: rows 0-1, cols 0-1
    A_blocks = {
        (0,0): {(0,0): {0*4+0: 1}, (0,1): {0*4+1: 1}, (1,0): {1*4+0: 1}, (1,1): {1*4+1: 1}},
        (0,1): {(0,0): {0*4+2: 1}, (0,1): {0*4+3: 1}, (1,0): {1*4+2: 1}, (1,1): {1*4+3: 1}},
        (1,0): {(0,0): {2*4+0: 1}, (0,1): {2*4+1: 1}, (1,0): {3*4+0: 1}, (1,1): {3*4+1: 1}},
        (1,1): {(0,0): {2*4+2: 1}, (0,1): {2*4+3: 1}, (1,0): {3*4+2: 1}, (1,1): {3*4+3: 1}},
    }
    
    # B block forms
    B_blocks = {
        (0,0): {(0,0): {0*9+(0+b_col_offset): 1}, (0,1): {0*9+(1+b_col_offset): 1}, (1,0): {1*9+(0+b_col_offset): 1}, (1,1): {1*9+(1+b_col_offset): 1}},
        (0,1): {(0,0): {0*9+(2+b_col_offset): 1}, (0,1): {0*9+(3+b_col_offset): 1}, (1,0): {1*9+(2+b_col_offset): 1}, (1,1): {1*9+(3+b_col_offset): 1}},
        (1,0): {(0,0): {2*9+(0+b_col_offset): 1}, (0,1): {2*9+(1+b_col_offset): 1}, (1,0): {3*9+(0+b_col_offset): 1}, (1,1): {3*9+(1+b_col_offset): 1}},
        (1,1): {(0,0): {2*9+(2+b_col_offset): 1}, (0,1): {2*9+(3+b_col_offset): 1}, (1,0): {3*9+(2+b_col_offset): 1}, (1,1): {3*9+(3+b_col_offset): 1}},
    }
    
    # C index function
    def C_idx(i, k):
        return (k + c_col_offset) * 4 + i
    
    all_products = []
    
    # Top-level Strassen: 7 products, each a 2×2 matrix product
    # P1 = (A00+A11) × (B00+B11)
    # P2 = (A10+A11) × B00
    # P3 = A00 × (B10-B11)
    # P4 = A11 × (B01-B10)
    # P5 = (A00+A10) × B11
    # P6 = (B00+B10) × (A10-A11)  -> A-form: (A10-A11), B-form: (B00+B10)
    # P7 = (B01+B11) × (A00-A10)  -> A-form: (A00-A10), B-form: (B01+B11)
    
    # For each top-level product, we have a 2×2 A-block and a 2×2 B-block
    # We need to combine the block forms into single linear forms for each 2×2 entry
    
    # Helper: combine 4 block forms into a single 2×2 set of linear forms
    def combine_blocks(blocks_list, op):
        """
        blocks_list: list of block dicts (each is {(0,0): form, (0,1): form, (1,0): form, (1,1): form})
        op: 'add' or 'sub'
        Returns a single block dict {(0,0): form, (0,1): form, (1,0): form, (1,1): form}
        """
        result = {}
        for pos in [(0,0), (0,1), (1,0), (1,1)]:
            if op == 'add':
                combined = blocks_list[0][pos]
                for b in blocks_list[1:]:
                    combined = _add_form(combined, b[pos])
            elif op == 'sub':
                combined = blocks_list[0][pos]
                for b in blocks_list[1:]:
                    combined = _sub_form(combined, b[pos])
            result[pos] = combined
        return result
    
    # P1: (A00+A11) × (B00+B11)
    A_P1 = combine_blocks([A_blocks[(0,0)], A_blocks[(1,1)]], 'add')
    B_P1 = combine_blocks([B_blocks[(0,0)], B_blocks[(1,1)]], 'add')
    all_products.extend(_strassen_2x2(A_P1, B_P1, C_idx))
    
    # P2: (A10+A11) × B00
    A_P2 = combine_blocks([A_blocks[(1,0)], A_blocks[(1,1)]], 'add')
    B_P2 = B_blocks[(0,0)]
    all_products.extend(_strassen_2x2(A_P2, B_P2, C_idx))
    
    # P3: A00 × (B10-B11)
    A_P3 = A_blocks[(0,0)]
    B_P3 = combine_blocks([B_blocks[(1,0)], B_blocks[(1,1)]], 'sub')
    all_products.extend(_strassen_2x2(A_P3, B_P3, C_idx))
    
    # P4: A11 × (B01-B10)
    A_P4 = A_blocks[(1,1)]
    B_P4 = combine_blocks([B_blocks[(0,1)], B_blocks[(1,0)]], 'sub')
    all_products.extend(_strassen_2x2(A_P4, B_P4, C_idx))
    
    # P5: (A00+A10) × B11
    A_P5 = combine_blocks([A_blocks[(0,0)], A_blocks[(1,0)]], 'add')
    B_P5 = B_blocks[(1,1)]
    all_products.extend(_strassen_2x2(A_P5, B_P5, C_idx))
    
    # P6: (B00+B10) × (A10-A11) -> A-form: (A10-A11), B-form: (B00+B10)
    A_P6 = combine_blocks([A_blocks[(1,0)], A_blocks[(1,1)]], 'sub')
    B_P6 = combine_blocks([B_blocks[(0,0)], B_blocks[(1,0)]], 'add')
    all_products.extend(_strassen_2x2(A_P6, B_P6, C_idx))
    
    # P7: (B01+B11) × (A00-A10) -> A-form: (A00-A10), B-form: (B01+B11)
    A_P7 = combine_blocks([A_blocks[(0,0)], A_blocks[(1,0)]], 'sub')
    B_P7 = combine_blocks([B_blocks[(0,1)], B_blocks[(1,1)]], 'add')
    all_products.extend(_strassen_2x2(A_P7, B_P7, C_idx))
    
    return all_products

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    all_products = []
    
    # First 4×4×4 block (B columns 0-3, C columns 0-3)
    all_products.extend(_build_4x4x4(0, 0))
    
    # Second 4×4×4 block (B columns 4-7, C columns 4-7)
    all_products.extend(_build_4x4x4(4, 4))
    
    # 4×4×1 part (B column 8, C column 8)
    for j in range(4):
        A_form = {}
        for i in range(4):
            A_form[i*4+j] = 1
        B_form = {j*9+8: 1}
        C_dict = {}
        for i in range(4):
            C_dict[8*4+i] = 1
        all_products.append((A_form, B_form, C_dict))
    
    rank = len(all_products)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (A_form, B_form, C_dict) in enumerate(all_products):
        for idx, coeff in A_form.items():
            U[idx][r] = coeff
        for idx, coeff in B_form.items():
            V[idx][r] = coeff
        for idx, coeff in C_dict.items():
            W[idx][r] = coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
