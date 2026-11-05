<<<<<<< SEARCH
        # Phase 5: Final polish with local search
        if time_left() > 3.0 and best_labels is not None:
            for _ in range(15):
                if time_left() < 1.5:
                    break
                labels_polish, cut_polish = local_search(best_labels, max_rounds=100)
                if cut_polish > best_cut:
                    best_cut = cut_polish
                    best_labels = labels_polish.copy()
                else:
                    break
=======
        # Phase 5: Final polish with local search
        if time_left() > 3.0 and best_labels is not None:
            for _ in range(30):
                if time_left() < 1.5:
                    break
                labels_polish, cut_polish = local_search(best_labels, max_rounds=500)
                if cut_polish > best_cut:
                    best_cut = cut_polish
                    best_labels = labels_polish.copy()
                else:
                    break
>>>>>>> REPLACE
