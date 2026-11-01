<<<<<<< SEARCH
    # Try to find rank-10 decomposition for 2x2x3
    U_223, V_223, W_223 = None, None, None
    if time.time() < deadline - 30:  # Leave some time margin
        result = _try_find_2x2x3_rank10(deadline - 10)
        if result is not None:
            U_223, V_223, W_223 = result
            # Verify
            if _verify_2x2x3(U_223, V_223, W_223):
                print(f"Found valid rank-10 decomposition for 2x2x3")
            else:
                U_223, V_223, W_223 = None, None, None
                print("Rank-10 decomposition failed verification")
    
    if U_223 is not None:
        # Use 6 blocks of 2x2 (Strassen, rank 7 each) + 1 block of 2x2x3 (rank 10)
        # Total rank = 6*7 + 10 = 52
        block_sizes = [2, 2, 2, 2, 2, 2, 3]
        num_blocks = len(block_sizes)
        total_rank = 6 * 7 + 10  # 52
    else:
        # Use 7 blocks of 2x2 (Strassen, rank 7 each) + 1 block of 2x1 (schoolbook, rank 4)
        # Total rank = 7*7 + 4 = 53
        block_sizes = [2, 2, 2, 2, 2, 2, 2, 1]
        num_blocks = len(block_sizes)
        total_rank = 7 * 7 + 4  # 53
=======
    # Try to find rank-10 decomposition for 2x2x3 (limited time)
    U_223, V_223, W_223 = None, None, None
    search_deadline = start_time + 60  # Give 60s for the search
    if time.time() < search_deadline:
        result = _try_find_2x2x3_rank10(search_deadline)
        if result is not None:
            U_223, V_223, W_223 = result
            if _verify_2x2x3(U_223, V_223, W_223):
                print(f"Found valid rank-10 decomposition for 2x2x3")
            else:
                U_223, V_223, W_223 = None, None, None
    
    if U_223 is not None:
        # Use 6 blocks of 2x2 (Strassen, rank 7 each) + 1 block of 2x2x3 (rank 10)
        # Total rank = 6*7 + 10 = 52
        block_sizes = [2, 2, 2, 2, 2, 2, 3]
        num_blocks = len(block_sizes)
        total_rank = 6 * 7 + 10  # 52
    else:
        # Use 7 blocks of 2x2 (Strassen, rank 7 each) + 1 block of 2x1 (schoolbook, rank 4)
        # Total rank = 7*7 + 4 = 53
        block_sizes = [2, 2, 2, 2, 2, 2, 2, 1]
        num_blocks = len(block_sizes)
        total_rank = 7 * 7 + 4  # 53
>>>>>>> REPLACE
