<<<<<<< SEARCH
    # Final polish
    if time.time() < deadline - 3:
        polished = polish(best_selected[:], deadline - 3)
        if len(polished) < best_size:
            best_size = len(polished)
            best_selected = polished[:]

    # Final local search pass
    if time.time() < deadline - 2:
        final_ls = local_search(best_selected[:], deadline - 2)
        if len(final_ls) < best_size:
            best_size = len(final_ls)
            best_selected = final_ls[:]

    best_words = [int_to_tuple(v) for v in best_selected]
    return best_words
=======
    # Additional strategy: Multiple local search restarts from best solution
    if time.time() < deadline - 10:
        for r in range(20):
            if time.time() > deadline - 5:
                break
            # Perturb the best solution
            perturbed = best_selected[:]
            # Remove a random codeword
            if perturbed:
                remove_idx = random.randint(0, len(perturbed) - 1)
                perturbed.pop(remove_idx)
            # Add a random codeword
            add_word = random.randint(0, total_space - 1)
            if add_word not in perturbed:
                perturbed.append(add_word)
            
            ls = local_search(perturbed[:], deadline - 5)
            if len(ls) < best_size:
                best_size = len(ls)
                best_selected = ls[:]

    # Final polish
    if time.time() < deadline - 3:
        polished = polish(best_selected[:], deadline - 3)
        if len(polished) < best_size:
            best_size = len(polished)
            best_selected = polished[:]

    # Final local search pass
    if time.time() < deadline - 2:
        final_ls = local_search(best_selected[:], deadline - 2)
        if len(final_ls) < best_size:
            best_size = len(final_ls)
            best_selected = final_ls[:]

    best_words = [int_to_tuple(v) for v in best_selected]
    return best_words
>>>>>>> REPLACE
