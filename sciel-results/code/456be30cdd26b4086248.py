<<<<<<< SEARCH
        moved_any_this_round = False

        # Strategy 1: Try to move each box from the last bin to any earlier bin
        box_order = list(range(len(bins[last_bin_idx])))
        box_order.sort(key=lambda idx: bins[last_bin_idx][idx][3] * bins[last_bin_idx][idx][4] * bins[last_bin_idx][idx][5])

        for box_idx in box_order:
            if time.time() > deadline - 15:
                break
            if box_idx >= len(bins[last_bin_idx]):
                continue
            for target_bin in range(last_bin_idx):
                if try_move_box(last_bin_idx, box_idx, target_bin):
                    moved_any_this_round = True
                    break
            if moved_any_this_round:
                break

        if moved_any_this_round:
            continue

        # Strategy 2: Try swaps between last bin and other bins
        for box_idx in range(len(bins[last_bin_idx])):
            if time.time() > deadline - 15:
                break
            for other_bin in range(last_bin_idx):
                for other_box_idx in range(len(bins[other_bin])):
                    if try_swap_boxes(last_bin_idx, box_idx, other_bin, other_box_idx):
                        moved_any_this_round = True
                        break
                if moved_any_this_round:
                    break
            if moved_any_this_round:
                break

        if not moved_any_this_round:
            break
=======
        moved_any_this_round = False

        # Strategy 1: Try to move each box from the last bin to any earlier bin
        box_order = list(range(len(bins[last_bin_idx])))
        box_order.sort(key=lambda idx: bins[last_bin_idx][idx][3] * bins[last_bin_idx][idx][4] * bins[last_bin_idx][idx][5])

        for box_idx in box_order:
            if time.time() > deadline - 15:
                break
            if box_idx >= len(bins[last_bin_idx]):
                continue
            for target_bin in range(last_bin_idx):
                if try_move_box(last_bin_idx, box_idx, target_bin):
                    moved_any_this_round = True
                    break
            if moved_any_this_round:
                break

        if moved_any_this_round:
            continue

        # Strategy 1b: Try relaxed support moves
        for box_idx in box_order:
            if time.time() > deadline - 15:
                break
            if box_idx >= len(bins[last_bin_idx]):
                continue
            for target_bin in range(last_bin_idx):
                if try_move_box_relaxed(last_bin_idx, box_idx, target_bin):
                    moved_any_this_round = True
                    break
            if moved_any_this_round:
                break

        if moved_any_this_round:
            continue

        # Strategy 2: Try swaps between last bin and other bins
        for box_idx in range(len(bins[last_bin_idx])):
            if time.time() > deadline - 15:
                break
            for other_bin in range(last_bin_idx):
                for other_box_idx in range(len(bins[other_bin])):
                    if try_swap_boxes(last_bin_idx, box_idx, other_bin, other_box_idx):
                        moved_any_this_round = True
                        break
                if moved_any_this_round:
                    break
            if moved_any_this_round:
                break

        if not moved_any_this_round:
            break
>>>>>>> REPLACE
