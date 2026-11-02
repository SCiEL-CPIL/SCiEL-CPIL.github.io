import time

DIMENSIONS = [3, 3, 8]

def construct():
    start_time = time.time()
    deadline = 140.0
    
    n, m, p = DIMENSIONS
    rank = 68
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term_complex(u_list, v_list, w_list):
        nonlocal col
        for idx, c in u_list:
            U[idx][col] += c
        for idx, c in v_list:
            V[idx][col] += c
        for idx, c in w_list:
            W[idx][col] += c
        col += 1

    for j in range(4):
        if time.time() > deadline:
            break
            
        c00 = 2*j*3
        c01 = (2*j+1)*3
        c10 = 2*j*3+1
        c11 = (2*j+1)*3+1
        c20 = 2*j*3+2
        c21 = (2*j+1)*3+2
        
        b00 = 2*j
        b01 = 2*j+1
        b10 = 8+2*j
        b11 = 8+2*j+1
        b20 = 16+2*j
        b21 = 16+2*j+1
        
        add_term_complex(
            [(0, 1), (4, 1)],
            [(b00, 1), (b01, 1), (b10, 1), (b11, 1)],
            [(c00, 1), (c11, 1)]
        )
        
        add_term_complex(
            [(3, 1), (4, 1)],
            [(b00, 1)],
            [(c10, 1), (c11, -1)]
        )
        
        add_term_complex(
            [(0, 1)],
            [(b10, 1), (b11, -1)],
            [(c01, 1), (c11, 1)]
        )
        
        add_term_complex(
            [(4, 1)],
            [(b00, 1), (b10, -1)],
            [(c00, 1), (c10, 1)]
        )
        
        add_term_complex(
            [(0, 1), (3, 1)],
            [(b11, 1)],
            [(c00, -1), (c01, 1)]
        )
        
        add_term_complex(
            [(4, 1), (0, -1)],
            [(b00, 1), (b10, 1)],
            [(c11, 1)]
        )
        
        add_term_complex(
            [(1, 1), (4, -1)],
            [(b10, 1), (b11, 1)],
            [(c00, 1), (c01, 1)]
        )
        
        add_term_complex([(2, 1)], [(b20, 1)], [(c00, 1)])
        add_term_complex([(2, 1)], [(b21, 1)], [(c01, 1)])
        add_term_complex([(5, 1)], [(b20, 1)], [(c10, 1)])
        add_term_complex([(5, 1)], [(b21, 1)], [(c11, 1)])
        
        add_term_complex([(6, 1)], [(b00, 1)], [(c20, 1)])
        add_term_complex([(6, 1)], [(b01, 1)], [(c21, 1)])
        add_term_complex([(7, 1)], [(b10, 1)], [(c20, 1)])
        add_term_complex([(7, 1)], [(b11, 1)], [(c21, 1)])
        
        add_term_complex([(8, 1)], [(b20, 1)], [(c20, 1)])
        add_term_complex([(8, 1)], [(b21, 1)], [(c21, 1)])

    return {"U": U, "V": V, "W": W}
