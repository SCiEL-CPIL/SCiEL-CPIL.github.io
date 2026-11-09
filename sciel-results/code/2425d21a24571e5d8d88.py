<<<<<<< SEARCH
    # Final shake local search
    if best_placements is not None and time.time() < deadline:
        rng2 = random.Random(999)
        best_placements, best_vol = _shake_local_search(best_placements, boxes, container, deadline, rng2, max_shakes=50)
=======
    # Final shake local search
    if best_placements is not None and time.time() < deadline:
        rng2 = random.Random(999)
        best_placements, best_vol = _shake_local_search(best_placements, boxes, container, deadline, rng2, max_shakes=80)
        
        # One more shake pass with different seed
        if time.time() < deadline:
            rng3 = random.Random(555)
            best_placements, best_vol = _shake_local_search(best_placements, boxes, container, deadline, rng3, max_shakes=40)
>>>>>>> REPLACE
